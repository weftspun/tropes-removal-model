# tropes-removal-model

ETNF parquet+zstd datalake, a hybrid trope detector (regex ONNX subgraph for
~23 mechanical tropes + a SetFit few-shot classifier for ~8 semantic ones,
merged into ONE ONNX model, plus a deterministic whole-document pass for 2
more tropes that are mechanical but need cross-sentence state -- see
runtime/cross_sentence.py) + a FLAN-T5-small rewriter, and a pre-commit/CI
gate that flags AI-writing tropes (per https://tropes.fyi/directory) with an
exact file/line span, the matching trope name + description, and a
suggested rewrite for every finding.

## Pipeline order

```
scripts/seed_tropes.py         -> seeds/trope.parquet (33 tropes, committed)
scripts/fetch_corpus.py        -> data/source_document.parquet, data/sentence.parquet (HC3, RAID, ai-text-detection-pile)
scripts/seed_labels.py         -> data/sentence_trope_label/ (weak regex labels, AND the regex patterns
                                   runtime/regex_onnx.py compiles into the shipped mechanical-trope detector)
scripts/synth_generate.py      -> same + data/sentence_rewrite/ (needs ANTHROPIC_API_KEY)
  or scripts/synth_seed_examples*.py (hand-authored fallback, no API key needed)
scripts/build_datalake.py      -> data/classifier_{train,val,test}.parquet, data/rewrite_pairs/
train_tropes.py                -> models/setfit_classifier/ (ONE multi-label SetFit model, ~8 semantic tropes only)
train_rewriter.py              -> models/rewriter/ (flan-t5-small seq2seq)
export_onnx_tropes.py          -> onnx_tropes/merged_model.onnx (regex + SetFit, ONE model), onnx_rewriter/
gate.py                        -> the CI/pre-commit entry point
```

`data/`, `models/`, `onnx_tropes/`, `onnx_rewriter/` are gitignored -- built
locally with a project venv (`.venv/`, Python 3.11) or restored from GitHub
Release assets by the workflows. `seeds/trope.parquet` is committed since
it's small and hand-authored.

## Design constraints (do not relax without re-reading the plan)

- Every datalake table key is a UUID, not an int surrogate; no JSON files
  anywhere in the lake -- parquet + zstd only.
- Every trope flag must resolve to one sentence's exact char span + the
  trope's name/category/description, never a bare document-level score.
- **Hybrid detection, not one model for all 33 tropes -- and not just a
  binary split.** The ~23 tropes that are lexical/structural/formatting
  patterns (Em-Dash Addiction, Delve and Friends, Tricolon Abuse, etc.) are
  caught by deterministic `RegexFullMatch` nodes (standard ONNX opset 20+,
  RE2 syntax, zero custom ops) compiled from `scripts/seed_labels.py`'s own
  patterns -- see `runtime/regex_onnx.py`, verified 0 mismatches vs Python's
  `re.search` across 30,000 real sentences. 2 more (Content Duplication,
  Historical Analogy Stacking) are also mechanical, not fuzzy, but need
  cross-sentence state a single-sentence regex can't hold ("does this exact
  text recur elsewhere in the document", "are there 2+ historical
  comparisons") -- these run as a plain deterministic whole-document pass in
  `runtime/cross_sentence.py`, called once per file from `gate.py`, not
  through the merged ONNX model. Only the remaining ~8 genuinely semantic
  tropes (False Vulnerability, Grandiose Stakes Inflation, Invented Concept
  Labels, etc.) go through a learned classifier -- no regex could ever catch
  these since there's no lexical tell to match. Mirrors how commercial
  writing-assistant products actually split this problem: parsers/rules for
  mechanical checks (single- or multi-sentence), ML only for fuzzy judgment
  calls. Lesson learned the hard way: two SetFit attempts at Content
  Duplication/Historical Analogy Stacking (single-sentence, then a
  windowed-context retry) both failed -- a pooled sentence embedding isn't
  the right representation for "does this recur elsewhere," and naive text
  windowing dilutes the very signal it's meant to add. Recognizing these as
  mechanical-but-document-scoped instead of reaching for a bigger model was
  the actual fix.
- **SetFit, not full fine-tuning, for the semantic classifier.** SetFit
  (contrastive sentence-transformer fine-tuning + a lightweight multi-label
  head, `multi_target_strategy="one-vs-rest"`) is purpose-built for
  few-shot text classification (its own docs target 8-16 examples/class);
  full end-to-end fine-tuning of a 66M-param backbone needs hundreds of
  examples per class to generalize instead of memorize. Verified on this
  repo's own data: 98.5% held-out accuracy from ~650 rows, 1 epoch.
  Backbone: `sentence-transformers/all-MiniLM-L6-v2` (same ONNX-reliable
  family the reference repo used for its embedding model). Use the
  **differentiable (torch) head**, not the default sklearn
  `LogisticRegression` head -- `skl2onnx`'s sklearn-head export produces
  float64 weights, which fails ONNX type-checking (`MatMul` dtype
  mismatch) against the sentence-transformer body's float32 output; an
  all-torch model traces to ONNX as consistently float32.
- **RegexFullMatch pattern portability (RE2, not Python `re`) --
  see `runtime/regex_onnx.py`'s `_to_re2_full_match` docstring for the
  full reasoning**: RE2 has no backreferences (`\1`) -- unroll them into
  explicit alternation instead; `.` doesn't match `\n` by default, so wrap
  only the outer `.*` anchors in `(?s:...)`, not the whole pattern (making
  it global broke a pattern whose own internal `.*` needs Python's default
  non-dotall behavior); always wrap the original pattern in a
  non-capturing group before adding `.*` anchors, since alternation
  precedence differs (`.*A|B.*` != `.*(A|B).*`).
- `merged_model.onnx`'s two branches (regex + semantic) both consume the
  same `text` input and each already output a full 33-column vector, zero
  in the columns they don't own; they're combined with a plain `Add` (no
  dynamic gather/scatter needed since exactly one branch is non-zero per
  column). See `export_onnx_tropes.py`.
- Must run locally on macOS via onnxruntime's CoreML EP (Metal/ANE);
  `runtime/infer.py` handles provider selection and logs which one was used.
- No GHA training workflow -- `train_tropes.py`, `train_rewriter.py`, and
  `export_onnx_tropes.py` are run locally (pixi env) and the resulting
  `onnx_tropes/`, `onnx_rewriter/` are published to a GitHub Release by hand
  for `tropes-gate.yml` to restore.
- Keep every parquet file under 100MB and the whole repo under 2GB.
  `sentence_trope_label`, `sentence_rewrite`, and `rewrite_pairs` are
  Hive-partitioned directories (one small file per trope_id/trope_name) via
  `runtime.datalake.PARTITION_COLUMNS` -- cheap because each row already
  belongs to exactly one trope. `sentence`, `source_document`, and the
  `classifier_{train,val,test}` wide tables are deliberately NOT partitioned
  by trope: a sentence has no single trope association (it may match zero,
  one, or several), so partitioning those would copy full sentence text
  into 33 partitions instead of one. `sentence.parquet` (~78MB) is the file
  to watch as more corpora get added -- it's the one closest to the ceiling.

## Running locally

```
python -m venv .venv && .venv/Scripts/activate  # or .venv/bin/activate on macOS/Linux
pip install pyarrow pandas datasets zstandard pytest
python scripts/seed_tropes.py
python scripts/fetch_corpus.py
python scripts/seed_labels.py
python scripts/synth_seed_examples.py
python scripts/build_datalake.py
pytest -q
```

Training (`train_tropes.py`, `train_rewriter.py`, `export_onnx_tropes.py`)
needs the heavier `pixi.toml` environment (setfit, torch, optimum,
onnxruntime-extensions) -- see `pixi install`.
