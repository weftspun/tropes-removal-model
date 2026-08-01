#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Scan all Hermes SKILL.md files for ASD-STE100 violations using
the tropes-removal-model's deterministic detectors.

Uses:
  - The same regex patterns as scripts/seed_labels.py (mechanical violations)
  - runtime/cross_sentence.py (paragraph-level, content duplication)
  - runtime/sentence_split.py (markdown-aware sentence splitting)

Reports findings grouped by file with exact line numbers and violation names.
"""
import os
import re
import sys
from pathlib import Path

# Add the tropes-removal-model root to sys.path
TROPES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TROPES_DIR))

# Import directly (these have no heavy dependencies)
from runtime.sentence_split import split_sentences
from runtime.cross_sentence import (
    CROSS_SENTENCE_VIOLATION_NAMES,
    detect_content_duplication,
    detect_historical_analogy_stacking,
    detect_long_paragraphs,
)

SKILLS_DIR = Path.home() / ".hermes" / "skills"
SKILLS_DIR = SKILLS_DIR.resolve()

# ---------------------------------------------------------------------------
# Regex patterns — verbatim from scripts/seed_labels.py
# These detect violations of ASD-STE100 writing rules via mechanical patterns.
# Not a complete STE checker — see https://asd-ste100.org for the full spec.
# ---------------------------------------------------------------------------

RIGHT_SINGLE_QUOTE = "\u2019"  # '
EM_DASH = "\u2014"             # —

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
        r"^.{0,10}$", re.M),  # placeholder; real detection is cross-sentence
}

VIOLATION_DESCRIPTIONS = {
    "Passive Voice": "STE Rule 1.5: Use active voice, not passive. 'is read' → 'read'.",
    "Semicolon Used": "STE Rule 8.3: Use a period, not a semicolon.",
    "Contraction Used": "STE Rule 2.2: Do not use contractions. Write the full form.",
    "Nominalization": "STE Rule 1.10: Use a verb, not a noun form. 'perform an analysis' → 'analyze'.",
    "Marketing Adjective": "Use plain descriptive words, not empty superlatives or marketing jargon.",
    "Phrasal Verb": "STE Rule 2.5: Use a single-word verb, not a phrasal verb. 'start' not 'spin up'.",
    "Banned Synonym": "Use the short common word. 'start' not 'commence'; 'use' not 'utilize'.",
    "Modal Hedge": "STE Rule 3.5: Do not use empty filler phrases. State commands and facts directly.",
    "-ing Main Verb": "STE Rule 1.5: Prefer simple tense over progressive form ('shows' not 'is showing').",
    "Stacked Auxiliaries": "STE Rule 1.5: Do not stack auxiliary verbs.",
    "Missing Article": "STE Rule 2.1: Use an article before a noun (a, an, the).",
    "Em-Dash Overuse": "STE Rule 8.1: Use standard punctuation. Em dashes for dramatic pauses are not technical writing.",
    "Three+ Nouns in a Row": "STE Rule 2.3: Do not use more than three consecutive nouns.",
    "Content Duplication": "Do not repeat the same content verbatim in the same document.",
    "Historical Analogy Stacking": "Use relevant examples only. Avoid rapid-fire historical analogies.",
    "Long Paragraph": "STE Rule 4.1: Maximum 6 sentences per paragraph.",
}

# Patterns for violations that read cleaner as regex-free counts
SENTENCE_TOO_LONG_PROC_WORDS = 20
SENTENCE_TOO_LONG_DESC_WORDS = 25

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_frontmatter(text):
    """Remove YAML frontmatter from skill text."""
    return FRONTMATTER_RE.sub("", text, count=1)


def find_all_skills():
    """Find all SKILL.md files under the skills directory."""
    skills = []
    for root, dirs, files in os.walk(SKILLS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "SKILL.md" in files:
            rel = Path(root).relative_to(SKILLS_DIR)
            skills.append((str(rel), Path(root) / "SKILL.md"))
    return sorted(skills)


def word_count(text):
    return len(text.split())


def scan_text(text, path_rel):
    """Scan text content for STE violations. Returns list of findings."""
    stripped = strip_frontmatter(text)
    if not stripped.strip():
        return []

    sentences = list(split_sentences(stripped))
    findings = []

    # --- 1. Mechanical regex violations ---
    for violation_name, pattern in PATTERNS.items():
        for s in sentences:
            match = pattern.search(s.text)
            if match:
                findings.append({
                    "name": violation_name,
                    "line": s.line,
                    "sentence": s.text[:120] + ("..." if len(s.text) > 120 else ""),
                    "match": match.group(0)[:60],
                })

    # --- 2. Sentence length violations ---
    for s in sentences:
        wc = word_count(s.text)
        if wc > SENTENCE_TOO_LONG_DESC_WORDS:
            label = "Sentence Too Long (descriptive)" if wc > SENTENCE_TOO_LONG_PROC_WORDS else "Sentence Too Long (procedural)"
            findings.append({
                "name": label,
                "line": s.line,
                "sentence": s.text[:120] + ("..." if len(s.text) > 120 else ""),
                "match": f"{wc} words",
            })

    # --- 3. Cross-sentence violations ---
    sentence_texts = [s.text for s in sentences]

    dupe_indices = detect_content_duplication(sentence_texts)
    for i in dupe_indices:
        findings.append({
            "name": "Content Duplication",
            "line": sentences[i].line,
            "sentence": sentences[i].text[:120] + ("..." if len(sentences[i].text) > 120 else ""),
            "match": "verbatim repeat",
        })

    analogy_indices = detect_historical_analogy_stacking(sentence_texts)
    for i in analogy_indices:
        findings.append({
            "name": "Historical Analogy Stacking",
            "line": sentences[i].line,
            "sentence": sentences[i].text[:120] + ("..." if len(sentences[i].text) > 120 else ""),
            "match": "historical analogy",
        })

    long_para_indices = detect_long_paragraphs(stripped, sentences)
    for i in long_para_indices:
        findings.append({
            "name": "Long Paragraph",
            "line": sentences[i].line,
            "sentence": sentences[i].text[:120] + ("..." if len(sentences[i].text) > 120 else ""),
            "match": "paragraph >6 sentences",
        })

    return findings


def main():
    skills = find_all_skills()
    print(f"Found {len(skills)} SKILL.md files", file=sys.stderr)

    violation_counts = {}
    total_skills_with_issues = 0

    for rel_path, full_path in skills:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        findings = scan_text(text, rel_path)
        if not findings:
            continue

        total_skills_with_issues += 1

        file_vtypes = {}
        for f in findings:
            file_vtypes.setdefault(f["name"], []).append(f)

        print(f"\n{'='*80}")
        print(f"📄 {rel_path}")
        print(f"{'='*80}")

        for vname in sorted(file_vtypes.keys()):
            hits = file_vtypes[vname]
            violation_counts[vname] = violation_counts.get(vname, 0) + len(hits)
            desc = VIOLATION_DESCRIPTIONS.get(vname, "")
            print(f"\n  ⚠  {vname} ({len(hits)} hit(s))")
            if desc:
                print(f"     {desc}")
            for h in hits[:8]:
                print(f"     L{h['line']}: \"{h['sentence']}\"")
            if len(hits) > 8:
                print(f"     ... and {len(hits) - 8} more")

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total skills scanned: {len(skills)}")
    print(f"Skills with violations: {total_skills_with_issues}")
    print(f"\nViolations by type:")
    for vname in sorted(violation_counts.keys(), key=lambda x: -violation_counts[x]):
        print(f"  {violation_counts[vname]:5d}  {vname}")

    print(f"\nTotal violations found: {sum(violation_counts.values())}")


if __name__ == "__main__":
    main()
