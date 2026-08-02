# ste-enforcer

A pre-commit / CI gate that flags violations of **ASD-STE100 Simplified Technical English** — the international specification for clear, unambiguous technical documentation — in your prose files, with an exact location, the violated rule name, and a suggested STE-compliant rewrite for every finding.

**Not a certified STE checker.** Full ASD-STE100 compliance requires human judgment (choosing the correct technical noun, assessing whether a sentence "makes good sense"). This tool covers the *mechanical* subset — passive voice, long sentences, nominalizations, marketing adjectives, phrasal verbs, banned synonyms, missing articles, stacked auxiliaries, semicolons, contractions, and paragraph limits — which is where most readability problems live. See the [free ASD-STE100 specification](https://asd-ste100.org) for the complete standard. Trademarks belong to ASD.

## Use it

```
pip install pre-commit  # or prek, a compatible Rust reimplementation
pre-commit install
```

`gate.py` reads the ONNX models from `onnx_violations/` and `onnx_rewriter/` (built locally and cached/restored from a GitHub Release by `ste-gate.yml`.
