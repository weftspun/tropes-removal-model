# tropes-removal-model

A pre-commit / CI gate that flags AI-writing "tropes" -- the recurring
stylistic tells cataloged at [tropes.fyi](https://tropes.fyi/directory)
(word choice like "delve", sentence patterns like Negative Parallelism,
formatting like Em-Dash Addiction, composition patterns like Fractal
Summaries) -- in your docs, with an exact location, the matching trope's
name and description, and a suggested rewrite for every finding.

## How it works

1. A datalake of real (HC3, RAID, ai-text-detection-pile) and
   hand-authored/synthetic text, normalized to Essential Tuple Normal Form
   and stored as zstd-compressed parquet with UUID keys throughout (see
   `runtime/datalake.py`).
2. A hybrid detector scores every sentence in a changed file: ~21
   lexical/structural/formatting tropes (Em-Dash Addiction, Delve and
   Friends, etc.) are caught by deterministic regex compiled directly into
   the ONNX graph (`runtime/regex_onnx.py`) -- zero training data, verified
   0 mismatches vs Python's `re.search` across 30,000 sentences. The
   remaining ~12 genuinely semantic tropes (False Vulnerability, Grandiose
   Stakes Inflation, etc.) go through a SetFit few-shot classifier, since
   there's no lexical tell to write a regex for.
3. A fine-tuned FLAN-T5-small model suggests a trope-free rewrite for every
   sentence that fires.
4. Both the detector and the rewriter are exported to ONNX -- the detector
   as a single merged model (regex branch + classifier branch, one
   `text` in, one 33-trope score vector out) -- so the gate itself needs no
   torch, and runs through onnxruntime's CoreML execution provider on macOS
   (Metal/ANE) or CPU elsewhere.

Every flag pinpoints one sentence -- file, line, character span -- and
names the trope and why, never a bare document-level score.

## Use it

```
pip install pre-commit  # or prek, a compatible Rust reimplementation
pre-commit install
```

`gate.py` reads the ONNX models from `onnx_tropes/` and `onnx_rewriter/`
(built locally and cached/restored from a GitHub Release by
`tropes-gate.yml`; see [CLAUDE.md](CLAUDE.md) for the full local build
pipeline).

## License

See [LICENSE](LICENSE).
