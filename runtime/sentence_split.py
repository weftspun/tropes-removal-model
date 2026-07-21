# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Deterministic sentence splitter shared by the training-data pipeline and
gate.py. Both sides must produce identical (char_start, char_end) spans for
the same text, or a trope flagged at train time won't line up with the span
gate.py reports at serve time.

Regex-based, dependency-free (no spacy/nltk model download) -- splits on
sentence-ending punctuation followed by whitespace and a capital/quote/digit,
while guarding the common abbreviations that would otherwise cause a false
split.
"""
import re
from dataclasses import dataclass

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "fig", "no", "vol", "approx", "inc", "corp", "co", "ltd", "st", "u.s",
    "u.k", "a.m", "p.m",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class Sentence:
    text: str
    char_start: int
    char_end: int
    line: int


def _line_of(offset: int, newline_offsets: list) -> int:
    """1-indexed line number for a char offset, via binary search over newline positions."""
    import bisect
    return bisect.bisect_right(newline_offsets, offset) + 1


def split_sentences(text: str) -> list:
    """Split text into Sentence spans with exact char offsets into `text`."""
    newline_offsets = [i for i, c in enumerate(text) if c == "\n"]

    boundaries = [0]
    for m in _SENTENCE_END.finditer(text):
        pos = m.start()
        preceding = text[:pos].rstrip()
        last_word = re.split(r"[\s(\[\"']+", preceding)[-1].lower().rstrip(".")
        if last_word in _ABBREV:
            continue
        boundaries.append(m.end())
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    sentences = []
    for start, end in zip(boundaries, boundaries[1:]):
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        s_off = start + raw.index(stripped)
        e_off = s_off + len(stripped)
        sentences.append(Sentence(
            text=stripped,
            char_start=s_off,
            char_end=e_off,
            line=_line_of(s_off, newline_offsets),
        ))
    return sentences
