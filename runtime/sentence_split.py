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

Markdown-aware on top of that: headers, list items, and blank lines are
treated as hard paragraph boundaries even without sentence-ending
punctuation (a markdown header or bullet often has none), and fenced code
blocks are skipped from scoring entirely -- code isn't prose, scoring it for
writing-style tropes is meaningless, and a large code block can otherwise
blow past a token classifier's own length limit as easily as any other
unbroken run of text. Discovered via a real ~319-word run-on "sentence" in
this repo's own CLAUDE.md: without this, an entire markdown section (a
paragraph + a fenced code block + surrounding prose, none of it separated by
sentence-ending punctuation) collapsed into one oversized unit that crashed
the ONNX classifier outright (its backbone's fixed position-embedding table
has no graceful truncation exposed at the ONNX level -- see gate.py's
MAX_SCORED_CHARS for the remaining defensive backstop). Splitting markdown
structure properly, not just capping length, is what keeps a long real
document's content actually analyzable instead of silently dropped past the
first chunk.
"""
import re
from dataclasses import dataclass

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "fig", "no", "vol", "approx", "inc", "corp", "co", "ltd", "st", "u.s",
    "u.k", "a.m", "p.m",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_HEADER = re.compile(r"^#{1,6}\s")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")


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


def _markdown_blocks(text: str) -> list:
    """Segment text into (start, end, is_code) line-based blocks at blank
    lines, ATX headers, list-item markers, and fenced code boundaries.
    Headers and list items are each their own single-line block (still
    eligible for further punctuation splitting); ordinary paragraph lines
    accumulate together until the next boundary; code-fence content is
    marked is_code=True so split_sentences() can skip it entirely."""
    lines = text.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    blocks = []
    paragraph_start = None
    in_code = False
    code_start = None

    def flush_paragraph(end):
        nonlocal paragraph_start
        if paragraph_start is not None and end > paragraph_start:
            blocks.append((paragraph_start, end, False))
        paragraph_start = None

    for i, line in enumerate(lines):
        start, end = line_starts[i], line_starts[i + 1]
        if _CODE_FENCE.match(line):
            if not in_code:
                flush_paragraph(start)
                in_code, code_start = True, start
            else:
                blocks.append((code_start, end, True))
                in_code = False
            continue
        if in_code:
            continue
        if not line.strip():
            flush_paragraph(start)
            continue
        if _HEADER.match(line) or _LIST_ITEM.match(line):
            flush_paragraph(start)
            blocks.append((start, end, False))
            continue
        if paragraph_start is None:
            paragraph_start = start

    if in_code and code_start is not None:
        blocks.append((code_start, line_starts[-1], True))
    else:
        flush_paragraph(line_starts[-1])
    return blocks


def _split_punctuation(text: str, base_offset: int, newline_offsets: list) -> list:
    """The original punctuation-based splitter, applied to one block of
    text; base_offset maps its local positions back into the full document."""
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
        s_off = base_offset + start + raw.index(stripped)
        e_off = s_off + len(stripped)
        sentences.append(Sentence(
            text=stripped, char_start=s_off, char_end=e_off,
            line=_line_of(s_off, newline_offsets),
        ))
    return sentences


def split_sentences(text: str) -> list:
    """Split text into Sentence spans with exact char offsets into `text`."""
    newline_offsets = [i for i, c in enumerate(text) if c == "\n"]
    sentences = []
    for start, end, is_code in _markdown_blocks(text):
        if is_code:
            continue
        sentences.extend(_split_punctuation(text[start:end], start, newline_offsets))
    return sentences
