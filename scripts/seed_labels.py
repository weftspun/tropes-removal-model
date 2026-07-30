# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""Weak-label sentences in `sentence` against per-violation regex/keyword
patterns derived from STE writing rules, writing sentence_trope_label rows
with label_source="seed-regex".

This is a TRAINING-DATA BOOTSTRAP ONLY. None of this regex logic ships in
gate.py -- the runtime gate always goes through the ONNX classifier. These
patterns exist only to give train_tropes.py a first pass of (noisy,
low-confidence) positive labels to bootstrap from, on top of the
higher-confidence synthetic pairs from scripts/synth_generate.py.

Only covers the mechanical violations that are matchable by pattern;
the purely semantic ones (word choice, safety-critical clarity, etc.)
get no seed-regex labels here and rely entirely on synthetic generation.

RE2 NOTE: These patterns are compiled into ONNX RegexFullMatch nodes by
runtime/regex_onnx.py. RE2 does NOT support \\u Unicode escapes (\\u2019,
\\u2014) -- use the literal characters instead. Python's re module handles
both forms, but the pattern string (.pattern) is what gets passed to RE2.
"""
import re
import sys
import uuid

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from runtime.datalake import SENTENCE_TROPE_LABEL_PATH, read_table, read_tropes, replace_rows

CONFIDENCE = 0.6  # weak/noisy signal; synthetic-gen labels are 1.0

# Unicode characters used literally (not as \uXXXX escapes) for RE2 compatibility
RIGHT_SINGLE_QUOTE = "\u2019"  # '
EM_DASH = "\u2014"              # —

# violation name -> compiled regex matched against sentence text (case-insensitive)
# These detect violations of ASD-STE100 writing rules via mechanical patterns.
# Not a complete STE checker -- see https://asd-ste100.org for the full spec.
PATTERNS = {
    "Passive Voice": re.compile(
        r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+ed|done|made|sent|read|built|"
        r"kept|held|set|put|run|written|shown|given|taken|found|got|gotten|seen|known|"
        r"thrown|drawn)\b", re.I),
    "Semicolon Used": re.compile(r";"),
    "Contraction Used": re.compile(
        r"\b\w+['" + RIGHT_SINGLE_QUOTE + r"](?:t|re|ve|ll|d|s|m)\b", re.I),
    "Nominalization": re.compile(
        r"\b(?:perform(?:s|ed)? an?|conduct(?:s|ed)? an?|provide(?:s|d)? a|carry out|"
        r"carries out|make use of|makes use of)\b", re.I),
    "Marketing Adjective": re.compile(
        r"\b(?:seamless|seamlessly|robust|powerful|cutting-edge|effortless|effortlessly|"
        r"world-class|next-generation|revolutionary|blazing|lightning-fast|elegant|"
        r"turnkey|state-of-the-art|game-changing|battle-tested|enterprise-grade|"
        r"supercharge|unlock|unleash)\b", re.I),
    "Phrasal Verb": re.compile(
        r"\b(?:spin up|spin down|reach out|dive into|dives into|diving into|"
        r"kick off|kicks off|roll out|rolls out|tear down|ramp up|circle back|"
        r"drill down|spun up|reaching out)\b", re.I),
    "Banned Synonym": re.compile(
        r"\b(?:begin|begins|commence|commences|initiate|initiates|originate|"
        r"utilize|utilizes|utilizing|leverage|leverages|leveraging|facilitate|facilitates|"
        r"ensure|ensures|ensuring|obtain|obtains|acquire|acquires|demonstrate|demonstrates|"
        r"additionally|furthermore|moreover|"
        r"utilization|aforementioned|henceforth|therein|"
        r"whilst|amongst|numerous|myriad|plethora|"
        r"prior to|subsequent to)\b", re.I),
    "Modal Hedge": re.compile(
        r"\b(?:it'?s? (?:is )?important to note|it should be noted|"
        r"it is worth noting|please note that|as mentioned|as noted above)\b", re.I),
    "-ing Main Verb": re.compile(
        r"\b(?:am|is|are|was|were)\s+\w+ing\b", re.I),
    "Stacked Auxiliaries": re.compile(
        r"\b(?:may help to|may be able to|might be able to|can help to|"
        r"could potentially|may possibly)\b", re.I),
    "Missing Article": re.compile(
        r"^\s*(?:Turn|Remove|Install|Open|Close|Press|Push|Pull|Lift|Lower|Insert|"
        r"Connect|Disconnect|Set|Check|Make|Adjust|Apply|Move|Rotate|Slide|Tighten|"
        r"Loosen|Replace|Clean|Fill|Drain)\s+[a-z]", re.M),
    "Em-Dash Overuse": re.compile(EM_DASH),
    "Three+ Nouns in a Row": re.compile(
        r"\b\w+\s+\w+\s+\w+\s+\w+\b", re.I),
    "Historical Analogy Stacking": re.compile(
        r"\b[A-Z][\w&' -]{1,40}\s+(?:did not|didn'?t)\s+(?:build|invent|create|start|make)\b"),
    "Content Duplication": re.compile(
        r"^.{0,10}$", re.M),  # placeholder; real detection is cross-sentence in runtime/cross_sentence.py
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

    replace_rows(SENTENCE_TROPE_LABEL_PATH, "label_source", "seed-regex", rows)
    print(f"wrote {len(rows)} weak seed-regex labels", file=sys.stderr)


if __name__ == "__main__":
    main()