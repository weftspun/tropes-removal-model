# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.cross_sentence import detect, detect_content_duplication, detect_historical_analogy_stacking


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


def test_detect_combines_both_checks():
    sentences = [
        "Netflix didn't build Blockbuster.",
        "Airbnb didn't build Marriott.",
        "The same sentence appears twice in this document.",
        "The same sentence appears twice in this document.",
    ]
    result = detect(sentences)
    assert result[0] == ["Historical Analogy Stacking"]
    assert result[1] == ["Historical Analogy Stacking"]
    assert result[2] == ["Content Duplication"]
    assert result[3] == ["Content Duplication"]
