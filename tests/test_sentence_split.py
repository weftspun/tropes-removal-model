# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.sentence_split import split_sentences


def test_offsets_roundtrip():
    text = "This is one sentence. This is another! And a third?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    for s in sentences:
        assert text[s.char_start:s.char_end] == s.text


def test_abbreviations_do_not_split():
    text = "Dr. Smith met Mr. Jones at 3 p.m. yesterday. They discussed the report."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].text.startswith("Dr. Smith")


def test_line_numbers():
    text = "First sentence here.\nSecond sentence here.\n\nThird one."
    sentences = split_sentences(text)
    assert [s.line for s in sentences] == [1, 2, 4]


def test_empty_text():
    assert split_sentences("") == []


def test_whitespace_only():
    assert split_sentences("   \n\n  ") == []


def test_markdown_header_is_its_own_sentence():
    text = "# My Title\n\nSome body text here."
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == ["# My Title", "Some body text here."]


def test_markdown_list_items_split_individually():
    text = "- First item without punctuation\n- Second item without punctuation\n"
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == [
        "- First item without punctuation",
        "- Second item without punctuation",
    ]


def test_fenced_code_block_excluded():
    text = "Some intro text.\n\n```\ncode line one\ncode line two\n```\n\nMore prose after."
    sentences = split_sentences(text)
    texts = [s.text for s in sentences]
    assert "code line one" not in " ".join(texts)
    assert "Some intro text." in texts
    assert "More prose after." in texts


def test_offsets_still_exact_with_markdown_structure():
    text = "# Header\n\n- Bullet one\n- Bullet two\n\nParagraph text ends here."
    sentences = split_sentences(text)
    for s in sentences:
        assert text[s.char_start:s.char_end] == s.text


def test_no_oversized_sentence_from_unpunctuated_markdown_block():
    # Regression test: a real markdown block (headers/bullets/code, none of
    # it ending in sentence punctuation) used to collapse into one giant
    # "sentence" long enough to crash the ONNX classifier's fixed-length
    # backbone -- see runtime/sentence_split.py's module docstring.
    text = (
        "# Title\n\n"
        "Some intro paragraph with no terminal punctuation issue here\n\n"
        "- item one about something\n"
        "- item two about something else\n"
        "- item three with more words describing another thing entirely\n\n"
        "```\nsome_code = 1\nmore_code = 2\n```\n\n"
        "Final paragraph.\n"
    )
    sentences = split_sentences(text)
    assert max(len(s.text.split()) for s in sentences) < 30
