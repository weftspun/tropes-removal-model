# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.cross_sentence import (
    detect, detect_content_duplication, detect_historical_analogy_stacking,
    detect_long_paragraphs, _split_paragraph_ranges,
)


def test_content_duplication_flags_verbatim_repeat():
    sentences = [
        "The quarterly report shows steady growth across every region.",
        "Something unrelated happened in between.",
        "The quarterly report shows steady growth across every region.",
    ]
    assert detect_content_duplication(sentences) == {0, 2}


def test_content_duplication_ignores_distinct_sentences():
    sentences = [
        "The quarterly report shows steady growth across every region.",
        "The dog ran across the yard and knocked over the flower pot.",
    ]
    assert detect_content_duplication(sentences) == set()


def test_content_duplication_ignores_short_boilerplate():
    sentences = ["Thanks.", "Some other content here.", "Thanks."]
    assert detect_content_duplication(sentences) == set()


def test_historical_analogy_needs_two_or_more():
    single = ["Netflix didn't build Blockbuster.", "Something else entirely."]
    assert detect_historical_analogy_stacking(single) == set()

    stacked = [
        "Netflix didn't build Blockbuster.",
        "Something else entirely.",
        "Airbnb didn't build Marriott.",
    ]
    assert detect_historical_analogy_stacking(stacked) == {0, 2}


def test_split_paragraph_ranges_blank_lines():
    text = "First paragraph here.\n\nSecond paragraph here."
    ranges = _split_paragraph_ranges(text)
    assert len(ranges) == 2


def test_split_paragraph_ranges_markdown_headings():
    text = "# Title\n\nSome intro text.\n\n## Section\n\nMore text here."
    ranges = _split_paragraph_ranges(text)
    assert len(ranges) >= 4


def test_split_paragraph_ranges_list_items():
    text = "Intro text.\n\n- First item\n- Second item\n\nOutro text."
    ranges = _split_paragraph_ranges(text)
    assert len(ranges) >= 3


def test_long_paragraph_not_flagged_for_short_paragraphs():
    """Each paragraph under 6 sentences should not be flagged."""
    from runtime.sentence_split import split_sentences
    text = "One sentence. Two sentence. Three sentence. Four sentence.\n\nNew paragraph. Two sentences here."
    sentences = list(split_sentences(text))
    flagged = detect_long_paragraphs(text, sentences)
    assert flagged == set()


def test_long_paragraph_flagged_for_over_six_sentences():
    """A paragraph with 7+ sentences should flag all its sentences."""
    from runtime.sentence_split import split_sentences
    # One paragraph with 7 sentences, no blank line break
    text = " ".join([f"Sentence number {i}." for i in range(1, 8)])
    sentences = list(split_sentences(text))
    flagged = detect_long_paragraphs(text, sentences)
    assert len(flagged) == 7


def test_long_paragraph_respects_blank_line_boundary():
    """Sentences in a short paragraph after a long one should not be flagged."""
    from runtime.sentence_split import split_sentences
    long_para = " ".join([f"Sentence {i}." for i in range(1, 8)])
    short_para = "Short paragraph. Only two sentences."
    text = long_para + "\n\n" + short_para
    sentences = list(split_sentences(text))
    flagged = detect_long_paragraphs(text, sentences)
    # Only the 7 sentences in the first paragraph should be flagged
    # The 2 sentences in the short paragraph should NOT be flagged
    short_flagged = {i for i in flagged if i >= 7}
    assert short_flagged == set()


def test_detect_combines_all_checks():
    from runtime.sentence_split import split_sentences
    text = (
        "Netflix didn't build Blockbuster.\n"
        "Airbnb didn't build Marriott.\n\n"
        "The same sentence appears twice in this document.\n"
        "The same sentence appears twice in this document.\n"
    )
    sentences = list(split_sentences(text))
    result = detect(text, sentences)
    # Historical Analogy Stacking should flag sentences 0 and 1
    assert "Historical Analogy Stacking" in result.get(0, [])
    assert "Historical Analogy Stacking" in result.get(1, [])
    # Content Duplication should flag sentences 2 and 3
    assert "Content Duplication" in result.get(2, [])
    assert "Content Duplication" in result.get(3, [])