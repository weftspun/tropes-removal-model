# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""Deterministic detection for STE violations that need cross-sentence or
document-level state:

  - Content Duplication: verbatim repeat of an earlier sentence in the same
    document. Detectable by comparing normalized sentence text.
  - Historical Analogy Stacking: two or more "X didn't build Y" sentences
    in the same document. A single instance is not stacking.
  - Long Paragraph: a paragraph exceeding 6 sentences. Paragraphs are
    delimited by blank lines (one or more consecutive blank lines), markdown
    headings (#..######), and list-item markers (- * + 1.). This is the
    industry-standard paragraph-boundary approach used by CommonMark
    renderers and text-processing libraries: blank-line splitting via the
    regex \\r?\\n\\s*\\r?\\n+ handles double newlines, Windows line endings,
    and extra blank lines; markdown structural elements (headings, list
    items) are also paragraph boundaries per the CommonMark spec.

The detect() function accepts the raw document text and the ordered sentence
list, so it can split paragraphs from the raw text and count sentences per
paragraph.
"""
import re

CROSS_SENTENCE_VIOLATION_NAMES = ["Content Duplication", "Historical Analogy Stacking",
                                   "Long Paragraph"]

_HISTORICAL_ANALOGY_PATTERN = re.compile(
    r"\b[A-Z][\w&' -]{1,40} (?:did not|didn'?t) (?:build|invent|create|start|make)\b")

_MIN_DUPLICATE_LEN = 25

# Industry-standard paragraph boundary: one or more blank lines.
# Matches \n\n, \r\n\r\n, \n\n\n, and lines with only whitespace between.
# This is the regex used by scikitplot's ParagraphChunker and CommonMark
# spec implementations for blank-line paragraph separation.
_PARA_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n+")

# Markdown structural elements that also start a new paragraph:
# ATX headings (#..######) and list-item markers (- * + 1. 2) etc.)
_MARKDOWN_BOUNDARY_RE = re.compile(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)", re.M)


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?")


def _split_paragraphs(raw_text):
    """Split raw text into paragraphs using blank-line boundaries and
    markdown structural elements.

    Returns a list of paragraph text strings.
    """
    # First split on blank lines (industry standard)
    parts = _PARA_SPLIT_RE.split(raw_text)
    # Then further split on markdown headings and list items
    paragraphs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on markdown structural boundaries
        sub_parts = _MARKDOWN_BOUNDARY_RE.split(part)
        for sub in sub_parts:
            sub = sub.strip()
            if sub:
                paragraphs.append(sub)
    return paragraphs


def _count_sentences_in_text(text):
    """Rough sentence count for a paragraph text: count sentence-ending
    punctuation followed by whitespace or end of string."""
    return len(re.findall(r"[.!?](?:\s|$)", text))


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


def detect_long_paragraphs(raw_text, sentences):
    """Flags sentences that belong to a paragraph exceeding 6 sentences.

    Uses the raw document text to split into paragraphs via blank-line and
    markdown structural boundaries, then counts sentences per paragraph.
    Each sentence's char_start is checked against paragraph boundaries to
    determine which paragraph it belongs to.

    sentences: list of Sentence objects with .text and .char_start
    raw_text: the original document text
    """
    if not raw_text or not sentences:
        return set()

    # Split raw text into paragraphs and compute char-offset ranges
    paragraphs = _split_paragraph_ranges(raw_text)
    if not paragraphs:
        return set()

    flagged = set()
    for para_start, para_end in paragraphs:
        # Count how many sentences fall within this paragraph
        para_sentence_indices = [
            i for i, s in enumerate(sentences)
            if para_start <= s.char_start < para_end
        ]
        if len(para_sentence_indices) > 6:
            flagged.update(para_sentence_indices)
    return flagged


def _split_paragraph_ranges(raw_text):
    """Split raw text into paragraphs and return (start_char, end_char) ranges.

    Uses blank-line splitting (\\r?\\n\\s*\\r?\\n+) plus markdown structural
    boundaries (headings, list items) as paragraph starters.
    """
    if not raw_text.strip():
        return []

    # Find all blank-line split points
    split_positions = [(m.start(), m.end()) for m in _PARA_SPLIT_RE.finditer(raw_text)]

    # Build paragraph ranges from split points
    ranges = []
    prev_end = 0
    for split_start, split_end in split_positions:
        if split_start > prev_end:
            ranges.append((prev_end, split_start))
        prev_end = split_end
    if prev_end < len(raw_text):
        ranges.append((prev_end, len(raw_text)))

    # Further split on markdown structural boundaries within each range
    refined_ranges = []
    for start, end in ranges:
        text_chunk = raw_text[start:end]
        # Find markdown boundary positions within this chunk
        boundaries = [(m.start(), m.end()) for m in _MARKDOWN_BOUNDARY_RE.finditer(text_chunk)]
        if not boundaries:
            refined_ranges.append((start, end))
            continue
        chunk_prev = 0
        for b_start, b_end in boundaries:
            if b_start > chunk_prev:
                refined_ranges.append((start + chunk_prev, start + b_start))
            chunk_prev = b_start
        if chunk_prev < len(text_chunk):
            refined_ranges.append((start + chunk_prev, end))

    # Filter out empty ranges
    return [(s, e) for s, e in refined_ranges if e > s]


def detect(raw_text, sentences):
    """Returns {sentence_index: [violation_name, ...]} for a whole document.

    raw_text: the original document text (for paragraph splitting)
    sentences: list of Sentence objects with .text and .char_start
    """
    sentence_texts = [s.text for s in sentences]
    result = {}
    for i in detect_content_duplication(sentence_texts):
        result.setdefault(i, []).append("Content Duplication")
    for i in detect_historical_analogy_stacking(sentence_texts):
        result.setdefault(i, []).append("Historical Analogy Stacking")
    for i in detect_long_paragraphs(raw_text, sentences):
        result.setdefault(i, []).append("Long Paragraph")
    return result