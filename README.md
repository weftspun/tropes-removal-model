# ste-enforcer

A pre-commit / CI gate that flags violations of **ASD-STE100 Simplified Technical English** — the international specification for clear, unambiguous technical documentation — in your prose files, with an exact location, the violated rule name, and a suggested STE-compliant rewrite for every finding.

**Not a certified STE checker.** Full ASD-STE100 compliance requires human judgment (choosing the correct technical noun, assessing whether a sentence "makes good sense"). This tool covers the *mechanical* subset — passive voice, long sentences, nominalizations, marketing adjectives, phrasal verbs, banned synonyms, missing articles, stacked auxiliaries, semicolons, contractions, and paragraph limits — which is where most readability problems live. See the [free ASD-STE100 specification](https://asd-ste100.org) for the complete standard. Trademarks belong to ASD.

## How it works

1. A datalake of real and synthetic text, normalized to Essential Tuple Normal Form and stored as zstd-compressed parquet with UUID keys throughout (see `runtime/datalake.py`).
2. A hybrid detector scores every sentence in a changed file: ~18 mechanical STE violations (passive voice, long sentences, semicolons, contractions, nominalizations, marketing adjectives, phrasal verbs, banned words, modal hedges, "-ing" main verbs, missing articles, stacked auxiliaries, no articles, etc.) are caught by deterministic regex compiled directly into the ONNX graph (`runtime/regex_onnx.py`) — zero training data. 2 more (content duplication, historical analogy stacking) are also mechanical but need cross-sentence state, so they run as a deterministic whole-document pass (`runtime/cross_sentence.py`). The remaining ~8 genuinely semantic violations (word-choice judgment, safety-critical phrasing, readability for non-native speakers) go through a SetFit few-shot classifier.
3. A fine-tuned FLAN-T5-small model suggests an STE-compliant rewrite for every sentence that fires.
4. Both the detector and the rewriter are exported to ONNX — the detector as a single merged model (regex branch + classifier branch, one `text` in, one violation-score vector out) — so the gate itself needs no torch, and runs through onnxruntime's CoreML execution provider on macOS (Metal/ANE) or CPU elsewhere.

Every flag pinpoints one sentence — file, line, character span — and names the STE rule and why, never a bare document-level score.

## Use it

```
pip install pre-commit  # or prek, a compatible Rust reimplementation
pre-commit install
```

`gate.py` reads the ONNX models from `onnx_violations/` and `onnx_rewriter/` (built locally and cached/restored from a GitHub Release by `ste-gate.yml`; see [CLAUDE.md](CLAUDE.md) for the full local build pipeline).

## STE violations detected

### Mechanical (regex — deterministic, zero training data needed)

| Category | What it catches |
|----------|----------------|
| Passive voice | "the file is read" → "read the file" |
| Sentence too long | >20 words (procedural), >25 words (descriptive) |
| Semicolons | Replace with period + new sentence |
| Contractions | "don't", "it's", "we'll" → "do not", "it is", "we will" |
| Nominalizations | "perform an analysis" → "analyze" |
| Marketing adjectives | "seamless", "robust", "powerful", etc. |
| Phrasal verbs | "spin up", "dive into", "kick off" |
| Banned synonyms | "utilize" → "use", "commence" → "start", "prior to" → "before" |
| Modal hedges | "it is important to note", "it should be noted" |
| "-ing" main verbs | "was running" → "ran" (when simple tense works) |
| Stacked auxiliaries | "may help to improve" → "improves" |
| Missing articles | "Turn shaft assembly" → "Turn the shaft assembly" |
| Em-dash overuse | Em dashes as dramatic pauses |
| Long paragraphs | >6 sentences per paragraph |

### Document-scoped mechanical (cross-sentence pass)

| Violation | What it catches |
|-----------|----------------|
| Content duplication | Verbatim repeated sections |
| Historical analogy stacking | ≥2 historical analogies in one document |

### Semantic (SetFit few-shot classifier — uses learned judgment)

| Violation | Description |
|-----------|-------------|
| Word-choice appropriateness | Is this the best STE-approved word for the concept? |
| Safety-critical clarity | Does the sentence warn or instruct with unambiguous urgency? |
| Readability for non-native speakers | Would a reader with limited English understand this? |
| Technical noun consistency | Is the same thing called by the same name throughout? |
| Sentence logic | Does the sentence "make good sense" in the STE sense? |

## License

See [LICENSE](LICENSE).