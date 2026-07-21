# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Weak-label sentences in `sentence` against per-trope regex/keyword seeds
derived directly from tropes.fyi's own example phrases, writing
sentence_trope_label rows with label_source="seed-regex".

This is a TRAINING-DATA BOOTSTRAP ONLY. Per the "route all 33 tropes through
the learned model" decision, none of this regex logic ships in gate.py --
the runtime gate always goes through the ONNX classifier. These patterns
exist only to give train_tropes.py a first pass of (noisy, low-confidence)
positive labels to bootstrap from, on top of the higher-confidence synthetic
pairs from scripts/synth_generate.py.

Only covers the lexical/structural tropes that are mechanically matchable;
the purely semantic tone/composition tropes (e.g. False Vulnerability,
Grandiose Stakes Inflation) get no seed-regex labels here and rely entirely
on synthetic generation for positive examples.
"""
import re
import sys
import uuid

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from runtime.datalake import SENTENCE_TROPE_LABEL_PATH, read_table, read_tropes, replace_rows

CONFIDENCE = 0.6  # weak/noisy signal; synthetic-gen labels are 1.0

# trope name -> compiled regex matched against sentence text (case-insensitive)
PATTERNS = {
    "Delve and Friends": re.compile(
        r"\b(delve|certainly!|utiliz|leverage|robust|streamline|harness(?:es|ing)?)\b", re.I),
    "Tapestry and Landscape": re.compile(
        r"\b(tapestry|landscape of|paradigm|synergy|ecosystem)\b", re.I),
    "The Serves As Dodge": re.compile(
        r"\b(serves as|stands as|marks (?:a|the))\b", re.I),
    "Negative Parallelism": re.compile(
        r"\bIt'?s not \b.{1,60}[.!?]\s*It'?s\b", re.I),
    "Not X. Not Y. Just Z.": re.compile(
        r"\bNot (?:a|an|\w+)\b.{1,40}\.\s*Not (?:a|an|\w+)\b.{1,40}\.\s*(?:Just|A)\b", re.I),
    "The X? A Y.": re.compile(
        r"\?\s*[A-Z][\w' ]{1,30}\.", re.I),
    # Backreferences (\1) work in Python's `re` but RE2 (used by ONNX's
    # RegexFullMatch, see runtime/regex_onnx.py) deliberately doesn't support
    # them -- this pattern is unrolled into explicit alternation instead of
    # \b(A|B|C)\b.*\b\1\b so the identical pattern object works in both engines.
    "Anaphora Abuse": re.compile(
        r"\bthey assume that\b.*\bthey assume that\b"
        r"|\bthe truth is\b.*\bthe truth is\b"
        r"|\bimagine a world\b.*\bimagine a world\b", re.I),
    "Tricolon Abuse": re.compile(
        r"\b\w+;\s*\w+\b.*\b\w+;\s*\w+\b", re.I),
    "It's Worth Noting": re.compile(
        r"\b(it'?s worth noting|importantly,|interestingly,|notably,)\b", re.I),
    "False Ranges": re.compile(
        r"\bfrom \w+ to \w+ to \w+\b", re.I),
    "Here's the Kicker": re.compile(
        r"\b(here'?s the kicker|here'?s the thing|here'?s where it gets interesting)\b", re.I),
    "Think of It As": re.compile(
        r"\bthink of it (?:as|like)\b", re.I),
    "Imagine a World Where": re.compile(
        r"\bimagine a world where\b", re.I),
    "The Truth Is Simple": re.compile(
        r"\bthe (?:truth|reality) is simpl", re.I),
    "Let's Break This Down": re.compile(
        r"\blet'?s break (?:this|it) down\b", re.I),
    "Vague Attributions": re.compile(
        r"\b(experts (?:say|argue|believe)|observers (?:say|note)|reports suggest)\b", re.I),
    "Em-Dash Addiction": re.compile(r"—"),
    "Bold-First Bullets": re.compile(r"^\s*[-*]\s+\*\*[^*]+\*\*", re.M),
    # Arrows only, not curly quotes/en-dash -- those are normal published-prose
    # typography and matching them was flooding this trope with false positives
    # (59,914 hits, mostly ordinary smart quotes, before this fix).
    "Unicode Decoration": re.compile(r"[→←↔⇒➜➔]"),
    "The Signposted Conclusion": re.compile(
        r"\b(in conclusion|to sum up|in summary)\b,?", re.I),
    "Despite Its Challenges": re.compile(
        r"\bdespite (?:its|these|the) challenges\b", re.I),
}

LABEL_NAMESPACE = uuid.UUID("8b3c8f4e-8f0a-4c2e-8e2f-2c9b5b7b6a44")


def main():
    tropes = read_tropes()
    name_to_id = dict(zip(tropes.column("name").to_pylist(), tropes.column("trope_id").to_pylist()))

    from runtime.datalake import SENTENCE_PATH
    sentences = read_table(SENTENCE_PATH)
    rows = []
    for sid, text in zip(sentences.column("sentence_id").to_pylist(), sentences.column("text").to_pylist()):
        for name, pattern in PATTERNS.items():
            if name not in name_to_id:
                continue
            if pattern.search(text):
                rows.append({
                    "sentence_id": sid,
                    "trope_id": name_to_id[name],
                    "label_source": "seed-regex",
                    "confidence": CONFIDENCE,
                })

    # Replace, not append: this pass is fully deterministic given the current
    # corpus + patterns, so re-running after a regex fix must drop the old
    # seed-regex rows rather than accumulate stale ones alongside the new.
    replace_rows(SENTENCE_TROPE_LABEL_PATH, "label_source", "seed-regex", rows)
    print(f"wrote {len(rows)} weak seed-regex labels", file=sys.stderr)


if __name__ == "__main__":
    main()
