# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Deterministic detection for the ~2 tropes that are neither a single-sentence
regex match (runtime/regex_onnx.py) nor a genuine fuzzy judgment call for the
SetFit classifier (train_tropes.py) -- they're mechanical, but the "unit of
work" is a whole document, not one sentence in isolation:

  - Content Duplication: "Repeating an entire section verbatim within the
    same piece." A same-document, word-for-word repeat is directly
    detectable by comparing normalized sentence text -- no model needed.
  - Historical Analogy Stacking: "Rapid-fire listing of historical companies
    or examples to manufacture false authority." A single "X didn't build Y"
    sentence isn't the trope; TWO OR MORE in the same document is. Also
    directly detectable: match the pattern per sentence, then require >=2
    matches in the document before flagging any of them.

Why not ONNX RegexFullMatch like the other 21 mechanical tropes: that op is
stateless per input string, with no way to compare one sentence against the
rest of the document it came from. Why not the SetFit classifier like the
other ~10 semantic tropes: two independent training runs (single-sentence,
then a windowed-context variant) both showed a single-sentence-embedding
model can't reliably learn "does this recur elsewhere" -- the windowed
variant's own held-out test numbers looked good, but it collapsed to
near-zero recall on realistic (non-synthetic-templated) documents, because
concatenating a text window dilutes the very signal a pooled sentence
embedding is supposed to isolate. A plain cross-sentence comparison sidesteps
both problems and needs no training data at all.

Called once per file from gate.py with the full ordered sentence list (not
per sentence), since both checks are inherently document-scoped.
"""
import re

CROSS_SENTENCE_TROPE_NAMES = ["Content Duplication", "Historical Analogy Stacking"]

# "Netflix didn't build Blockbuster." / "Apple did not invent the phone."
_HISTORICAL_ANALOGY_PATTERN = re.compile(
    r"\b[A-Z][\w&' -]{1,40} (?:did not|didn'?t) (?:build|invent|create|start|make)\b")

# Below this length, exact-repeat matches are usually short filler/boilerplate
# ("Thanks.", "Yes.") rather than a genuinely duplicated section -- excluding
# them avoids false positives on trivially common short sentences.
_MIN_DUPLICATE_LEN = 25


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?")


def detect_content_duplication(sentences):
    """sentences: ordered list of sentence text strings for one document.
    Returns the set of indices whose text is a verbatim (normalized) repeat
    of an earlier sentence in the same document."""
    seen = {}
    flagged = set()
    for i, text in enumerate(sentences):
        norm = _normalize(text)
        if len(norm) < _MIN_DUPLICATE_LEN:
            continue
        if norm in seen:
            flagged.add(i)
            flagged.add(seen[norm])
        else:
            seen[norm] = i
    return flagged


def detect_historical_analogy_stacking(sentences):
    """Flags every matching sentence, but only if the document contains 2 or
    more -- a single "X didn't build Y" is just a sentence, not stacking."""
    matches = [i for i, text in enumerate(sentences) if _HISTORICAL_ANALOGY_PATTERN.search(text)]
    return set(matches) if len(matches) >= 2 else set()


def detect(sentences):
    """Returns {sentence_index: [trope_name, ...]} for a whole document's
    ordered sentence text list."""
    result = {}
    for i in detect_content_duplication(sentences):
        result.setdefault(i, []).append("Content Duplication")
    for i in detect_historical_analogy_stacking(sentences):
        result.setdefault(i, []).append("Historical Analogy Stacking")
    return result
