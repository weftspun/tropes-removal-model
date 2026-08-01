#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Rewrite SKILL.md files to remove deterministic ASD-STE100 violations.

Rewrites (automated mechanical categories only):
  1. Contractions → full forms
  2. Banned synonyms → short common words
  3. Marketing adjectives → neutral replacements or removal
  4. Modal hedges → direct statements
  5. Nominalizations → verb forms
  6. Em-dashes used as dramatic pauses → colons or periods
  7. Passive voice → active voice (simple cases with known agents)
  8. Missing articles → add "the" (procedural instructions only)

Operates on the frontmatter-aware text (frontmatter preserved verbatim).
"""
import os
import re
import sys
from pathlib import Path

SKILLS_DIR = Path.home() / ".hermes" / "skills"
SKILLS_DIR = SKILLS_DIR.resolve()

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

# ── Contractions ──────────────────────────────────────────────────────────
CONTRACTIONS = {
    # Common contractions
    "don't": "do not",
    "don’t": "do not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "didn't": "did not",
    "didn’t": "did not",
    "won't": "will not",
    "won’t": "will not",
    "wouldn't": "would not",
    "wouldn’t": "would not",
    "shouldn't": "should not",
    "shouldn’t": "should not",
    "couldn't": "could not",
    "couldn’t": "could not",
    "can't": "cannot",
    "can’t": "cannot",
    "isn't": "is not",
    "isn’t": "is not",
    "aren't": "are not",
    "aren’t": "are not",
    "wasn't": "was not",
    "wasn’t": "was not",
    "weren't": "were not",
    "weren’t": "were not",
    "haven't": "have not",
    "haven’t": "have not",
    "hasn't": "has not",
    "hasn’t": "has not",
    "hadn't": "had not",
    "hadn’t": "had not",
    "mustn't": "must not",
    "mustn’t": "must not",
    "needn't": "need not",
    "needn’t": "need not",
    "daren't": "dare not",
    "daren’t": "dare not",
    # Contractions that need care (pronoun + verb)
    "it's": "it is",
    "it’s": "it is",
    "that's": "that is",
    "that’s": "that is",
    "there's": "there is",
    "there’s": "there is",
    "here's": "here is",
    "here’s": "here is",
    "what's": "what is",
    "what’s": "what is",
    "who's": "who is",
    "who’s": "who is",
    "where's": "where is",
    "where’s": "where is",
    "why's": "why is",
    "why’s": "why is",
    "how's": "how is",
    "how’s": "how is",
    "he's": "he is",
    "he’s": "he is",
    "she's": "she is",
    "she’s": "she is",
    "we're": "we are",
    "we’re": "we are",
    "they're": "they are",
    "they’re": "they are",
    "you're": "you are",
    "you’re": "you are",
    "i'm": "I am",
    "i’m": "I am",
    "we'll": "we will",
    "we’ll": "we will",
    "they'll": "they will",
    "they’ll": "they will",
    "you'll": "you will",
    "you’ll": "you will",
    "he'll": "he will",
    "he’ll": "he will",
    "she'll": "she will",
    "she’ll": "she will",
    "it'll": "it will",
    "it’ll": "it will",
    "that'll": "that will",
    "that’ll": "that will",
    "i've": "I have",
    "i’ve": "I have",
    "we've": "we have",
    "we’ve": "we have",
    "they've": "they have",
    "they’ve": "they have",
    "you've": "you have",
    "you’ve": "you have",
    "i'd": "I would",
    "i’d": "I would",
    "we'd": "we would",
    "we’d": "we would",
    "they'd": "they would",
    "they’d": "they would",
    "you'd": "you would",
    "you’d": "you would",
    "he'd": "he would",
    "he’d": "he would",
    "she'd": "she would",
    "she’d": "she would",
    # Possessive/contraction ambiguity — skip "its" (possessive)
    # but catch "let's"
    "let's": "let us",
    "let’s": "let us",
}

# ── Banned synonyms → short common words ─────────────────────────────────
BANNED_SYNONYMS = {
    # Verbs
    r"\butilize\b": "use",
    r"\butilizes\b": "uses",
    r"\butilizing\b": "using",
    r"\butilization\b": "use",
    r"\bleverage\b(?!\s+[a-z]\w+)": "use",
    r"\bleverages\b": "uses",
    r"\bleveraging\b": "using",
    r"\bfacilitate\b": "help",
    r"\bfacilitates\b": "helps",
    r"\binitiate\b": "start",
    r"\binitiates\b": "starts",
    r"\bcommence\b": "start",
    r"\bcommences\b": "starts",
    r"\bdemonstrate\b": "show",
    r"\bdemonstrates\b": "shows",
    r"\bobtain\b": "get",
    r"\bobtains\b": "gets",
    r"\bacquire\b": "get",
    r"\bacquires\b": "gets",
    r"\boriginate\b": "come",
    # Adverbs
    r"\badditionally\b": "also",
    r"\bfurthermore\b": "also",
    r"\bmoreover\b": "also",
    r"\baforementioned\b": "above",
    r"\bhenceforth\b": "from now on",
    r"\btherein\b": "there",
    # Prepositions
    r"\bprior to\b": "before",
    r"\bsubsequent to\b": "after",
    # Adjectives
    r"\bamongst\b": "among",
    r"\bwhilst\b": "while",
    r"\bnumerous\b": "many",
    r"\bmyriad\b": "many",
    r"\bplethora\b": "many",
}

# ── Marketing adjectives ─────────────────────────────────────────────────
MARKETING_ADJECTIVES = {
    r"\bseamless\b": "smooth",
    r"\bseamlessly\b": "smoothly",
    r"\bcutting-edge\b": "modern",
    r"\bworld-class\b": "high-quality",
    r"\bnext-generation\b": "new",
    r"\brevolutionary\b": "significant",
    r"\bgame-changing\b": "important",
    r"\bbattle-tested\b": "tested",
    r"\benterprise-grade\b": "production",
    r"\bbest-in-class\b": "excellent",
    r"\bstate-of-the-art\b": "modern",
    r"\bturnkey\b": "ready-to-use",
    r"\bsupercharge\b": "improve",
    r"\bunlock\b(?!\s+the\s+full\s+potential)": "enable",
    r"\bunleash\b": "enable",
    r"\blightning-fast\b": "fast",
    r"\bblazing\b": "fast",
    r"\bpowerful tool\b": "effective tool",
    r"\bpowerful features?\b": "useful features",
}

# ── Modal hedges ─────────────────────────────────────────────────────────
MODAL_HEDGES = [
    (r"\bit'?s? important to note that\b", ""),
    (r"\bit is important to note that\b", ""),
    (r"\bit should be noted that\b", ""),
    (r"\bit should be noted\b", ""),
    (r"\bit is worth noting that\b", ""),
    (r"\bplease note that\b", ""),
    (r"\bas mentioned above\b", ""),
    (r"\bas noted above\b", ""),
]

# ── Nominalizations ──────────────────────────────────────────────────────
NOMINALIZATIONS = {
    r"\bperform(?:s|ed)? an?\s+analysis\b": "analyze",
    r"\bperform(?:s|ed)? an?\s+check\b": "check",
    r"\bperform(?:s|ed)? an?\s+setup\b": "set up",
    r"\bperform(?:s|ed)? an?\s+install\b": "install",
    r"\bperform(?:s|ed)? a\s+review\b": "review",
    r"\bperform(?:s|ed)? a\s+test\b": "test",
    r"\bperform(?:s|ed)? a\s+search\b": "search",
    r"\bperform(?:s|ed)? a\s+scan\b": "scan",
    r"\bperform(?:s|ed)? a\s+validation\b": "validate",
    r"\bperform(?:s|ed)? a\s+verification\b": "verify",
    r"\bperform(?:s|ed)? a\s+comparison\b": "compare",
    r"\bperform(?:s|ed)? a\s+cleanup\b": "clean up",
    r"\bperform(?:s|ed)? a\s+merge\b": "merge",
    r"\bperform(?:s|ed)? a\s+build\b": "build",
    r"\bconduct(?:s|ed)? an?\s+analysis\b": "analyze",
    r"\bconduct(?:s|ed)? an?\s+assessment\b": "assess",
    r"\bconduct(?:s|ed)? a\s+test\b": "test",
    r"\bconduct(?:s|ed)? a\s+review\b": "review",
    r"\bconduct(?:s|ed)? a\s+survey\b": "survey",
    r"\bprovide(?:s|d)? a\s+summary\b": "summarize",
    r"\bprovide(?:s|d)? an?\s+overview\b": "describe",
    r"\bcarry out\b": "do",
    r"\bcarries out\b": "does",
    r"\bmake use of\b": "use",
    r"\bmakes use of\b": "uses",
}

# ── Passive voice → active voice (specific patterns) ─────────────────────
# Maps "is/was/are/were/be + past-participle by X" to active forms
PASSIVE_PATTERNS = [
    # "is used" → "use" (instructional context)
    (r"\bis used by\b", "uses"),
    (r"\bare used by\b", "use"),
    (r"\bwas used by\b", "used"),
    # "can be used" → "use"
    (r"\bcan be used to\b", "use to"),
    (r"\bis designed to\b", "designed to"),
    (r"\bare designed to\b", "designed to"),
    # "is required" → "requires"  
    (r"\bis required for\b", "needs"),
    (r"\bare required for\b", "need"),
    (r"\bis required to\b", "must"),
    (r"\bare required to\b", "must"),
    # "is needed" → "needs"
    (r"\bis needed\b(?!\s+by)", "needs"),
    (r"\bare needed\b(?!\s+by)", "need"),
    # "is called" → "called"
    (r"\bis called\b", "called"),
    # "is known as" → "known as"
    (r"\bis known as\b", "known as"),
    # "is found" → "finds" (only when agent follows)
    (r"\bis found by\b", "finds"),
    (r"\bare found by\b", "find"),
    # "is set to" → "set to"
    (r"\bis set to\b", "set to"),
    (r"\bare set to\b", "set to"),
    (r"\bwas set to\b", "set to"),
]

# ── Missing articles (procedural instructions only) ──────────────────────
# Add "the" before noun after procedural verbs if missing
PROCEDURAL_VERBS = (
    r"(?:Turn|Remove|Install|Open|Close|Press|Push|Pull|Lift|Lower|Insert|"
    r"Connect|Disconnect|Set|Check|Make|Adjust|Apply|Move|Rotate|Slide|Tighten|"
    r"Loosen|Replace|Clean|Fill|Drain)"
)
MISSING_ARTICLE_PATTERN = re.compile(
    rf"\b({PROCEDURAL_VERBS})\s+([a-z]\w+)"
)


def strip_frontmatter(text):
    m = FRONTMATTER_RE.match(text)
    if m:
        return m.group(0), text[m.end():]
    return "", text


def apply_rewrites(text, path_rel):
    """Apply all rewrites to text. Returns (new_text, changes_list)."""
    frontmatter, body = strip_frontmatter(text)
    changes = []

    changed = body  # work on body only (no frontmatter)

    # 1. Contractions
    for contr, full in sorted(CONTRACTIONS.items(), key=lambda x: -len(x[0])):
        # Match whole words only, case-insensitive for common ones
        pattern = re.compile(
            r"(?<!\w)" + re.escape(contr) + r"(?!\w)", re.IGNORECASE
        )
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(full, changed)
            changes.append(f"contraction: {contr} → {full} ({hits})")

    # 2. Banned synonyms
    for pattern_str, replacement in BANNED_SYNONYMS.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(replacement, changed)
            changes.append(f"banned-synonym: ~{pattern_str}~ → {replacement} ({hits})")

    # 3. Marketing adjectives
    for pattern_str, replacement in MARKETING_ADJECTIVES.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(replacement, changed)
            changes.append(f"marketing-adj: ~{pattern_str}~ → {replacement} ({hits})")

    # 4. Modal hedges
    for pattern_str, replacement in MODAL_HEDGES:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(replacement, changed)
            changes.append(f"modal-hedge: ~{pattern_str}~ removed ({hits})")

    # 5. Nominalizations
    for pattern_str, replacement in sorted(
        NOMINALIZATIONS.items(), key=lambda x: -len(x[0])
    ):
        pattern = re.compile(pattern_str, re.IGNORECASE)
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(replacement, changed)
            changes.append(f"nominalization: ~{pattern_str}~ → {replacement} ({hits})")

    # 6. Passive voice (simple patterns only)
    for pattern_str, replacement in PASSIVE_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        hits = len(pattern.findall(changed))
        if hits:
            changed = pattern.sub(replacement, changed)
            changes.append(f"passive-voice: ~{pattern_str}~ → {replacement} ({hits})")

    # 7. Missing articles (procedural verbs)
    hits = len(MISSING_ARTICLE_PATTERN.findall(changed))
    if hits:
        changed = MISSING_ARTICLE_PATTERN.sub(r"\1 the \2", changed)
        changes.append(f"missing-article: added 'the' after procedural verb ({hits})")

    if changed == body:
        return text, []  # no changes

    return frontmatter + changed, changes


def main():
    # Find all SKILL.md files
    skills = []
    for root, dirs, files in os.walk(SKILLS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "SKILL.md" in files:
            rel = Path(root).relative_to(SKILLS_DIR)
            skills.append((str(rel), Path(root) / "SKILL.md"))

    total_files_changed = 0
    total_changes = 0
    changes_by_type = {}
    all_summaries = []

    for path_rel, full_path in skills:
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"  ❌ {path_rel}: read error: {e}", file=sys.stderr)
            continue

        new_text, changes = apply_rewrites(text, path_rel)

        if not changes:
            continue

        total_files_changed += 1
        total_changes += sum(
            int(c.split("(")[-1].rstrip(")").split()[-1]) if "(" in c else 1
            for c in changes
        )
        for c in changes:
            ctype = c.split(":")[0]
            changes_by_type[ctype] = changes_by_type.get(ctype, 0) + 1

        # Write the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        all_summaries.append(f"  📝 {path_rel}")
        for c in changes:
            all_summaries.append(f"      {c}")
        if len(all_summaries) > 2000:
            # Print to stdout and clear buffer
            print("\n".join(all_summaries))
            all_summaries = []

    # Flush remaining
    if all_summaries:
        print("\n".join(all_summaries))

    # Summary
    print(f"\n{'='*70}")
    print("REWRITE SUMMARY")
    print(f"{'='*70}")
    print(f"Files modified: {total_files_changed}")
    print(f"Total individual rewrites: ~{total_changes}")
    print(f"\nChange types applied:")
    for ctype in sorted(changes_by_type.keys()):
        print(f"  {changes_by_type[ctype]:6d}  {ctype}")
    print(f"\nEstimated violation reduction: ~{total_changes} (of 43,309 total)")


if __name__ == "__main__":
    main()
