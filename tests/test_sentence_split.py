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
