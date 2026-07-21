# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Write seeds/trope.parquet: the 33 AI-writing tropes cataloged at
https://tropes.fyi/directory, normalized to the `trope` relation
(trope_id UUID, name, category, description, example_phrase).

trope_id is a UUIDv5 derived from the trope name (fixed namespace), so
re-running this script is idempotent and reproducible -- never uuid4.
No JSON is written; this script is the only place the catalog is
hand-authored, directly as parquet rows.
"""
import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

NAMESPACE = uuid.UUID("d9f6a9f2-9d0a-4c2e-8e2f-2c9b5b7b6a11")  # fixed; do not change
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seeds", "trope.parquet")

# (name, category, description, example_phrase)
TROPES = [
    ("Quietly and Other Magic Adverbs", "word-choice",
     "Overuse of adverbs conveying subtle importance without evidence (quietly, deeply, fundamentally, remarkably).",
     "quietly orchestrating workflows, decisions, and interactions"),
    ("Delve and Friends", "word-choice",
     "Overused vocabulary that signals AI authorship (delve, certainly, utilize, leverage, robust, streamline, harness).",
     "Let's delve into the details..."),
    ("Tapestry and Landscape", "word-choice",
     "Ornate, grandiose nouns replacing simpler words (tapestry, landscape, paradigm, synergy, ecosystem).",
     "The rich tapestry of human experience..."),
    ("The Serves As Dodge", "word-choice",
     "Replacing a plain \"is/are\" with a pompous alternative (serves as, stands as, marks).",
     "The building serves as a reminder of the city's heritage."),
    ("Negative Parallelism", "sentence-structure",
     "The \"It's not X -- it's Y\" pattern that frames an ordinary statement as a surprising reframe.",
     "It's not bold. It's backwards."),
    ("Not X. Not Y. Just Z.", "sentence-structure",
     "A dramatic countdown that negates several options before revealing the actual point.",
     "Not a bug. Not a feature. A fundamental design flaw."),
    ("The X? A Y.", "sentence-structure",
     "A self-posed rhetorical question answered immediately for manufactured dramatic effect.",
     "The result? Devastating."),
    ("Anaphora Abuse", "sentence-structure",
     "Repeating the same sentence opening multiple times in succession.",
     "They assume that users will pay... They assume that..."),
    ("Tricolon Abuse", "sentence-structure",
     "Overuse of rule-of-three patterns, often stretched to four or five items.",
     "Products impress; platforms empower. Products solve; platforms create."),
    ("It's Worth Noting", "sentence-structure",
     "Filler transitions that signal nothing (notably, importantly, interestingly, it's worth noting that).",
     "It's worth noting that this approach has limitations."),
    ("Superficial Analyses", "sentence-structure",
     "Tacking a present-participle phrase onto a sentence to inject shallow, unearned significance.",
     "contributing to the region's rich cultural heritage"),
    ("False Ranges", "sentence-structure",
     "\"From X to Y\" constructions where the items aren't on any real spectrum or scale.",
     "From innovation to implementation to cultural transformation."),
    ("Short Punchy Fragments", "paragraph-structure",
     "Excessive standalone sentence fragments used for manufactured emphasis.",
     "He published this. Openly. In a book. As a priest."),
    ("Listicle in a Trench Coat", "paragraph-structure",
     "A numbered list disguised as continuous prose via \"The first... The second...\" enumeration.",
     "The first wall is... The second wall is..."),
    ("Here's the Kicker", "tone",
     "False-suspense transitions promising drama before a mundane observation.",
     "Here's the kicker."),
    ("Think of It As", "tone",
     "A patronizing analogy that assumes the reader needs a metaphor to understand a plain concept.",
     "Think of it like a highway system for data."),
    ("Imagine a World Where", "tone",
     "Futuristic framing that lists wonderful outcomes contingent on accepting an unproven premise.",
     "Imagine a world where every tool you use has quiet, invisible intelligence."),
    ("False Vulnerability", "tone",
     "Performative self-awareness that simulates authenticity while remaining polished and safe.",
     "And yes, I'm openly in love with the platform model."),
    ("The Truth Is Simple", "tone",
     "Asserting obviousness in place of proving the claim, dismissing prior arguments by fiat.",
     "The reality is simpler and less flattering."),
    ("Grandiose Stakes Inflation", "tone",
     "Inflating an ordinary argument to world-historical significance.",
     "This will fundamentally reshape how we think about everything."),
    ("Let's Break This Down", "tone",
     "A pedagogical voice that assumes an expert audience needs elementary hand-holding.",
     "Let's break this down step by step."),
    ("Vague Attributions", "tone",
     "Invoking unnamed \"experts,\" \"observers,\" or \"reports\" without any specific citation.",
     "Experts argue that this approach has significant drawbacks."),
    ("Invented Concept Labels", "tone",
     "Coining a compound term that sounds analytical but is undefined outside this text.",
     "the supervision paradox"),
    ("Em-Dash Addiction", "formatting",
     "Compulsive overuse of em dashes for dramatic pauses and asides.",
     "The problem -- and this is the part nobody talks about -- is scale."),
    ("Bold-First Bullets", "formatting",
     "Every bullet point begins with a bolded phrase, a telltale AI markdown pattern.",
     "**Speed:** Every single bullet point begins with a bold keyword."),
    ("Unicode Decoration", "formatting",
     "Unnatural use of arrows, smart quotes, or other special characters.",
     "Input -> Processing -> Output"),
    ("Fractal Summaries", "composition",
     "Restating the same points at every structural level: intro preview, section recap, conclusion.",
     "In this section, we'll explore... [later] ...as we've seen..."),
    ("The Dead Metaphor", "composition",
     "A single metaphor repeated five to ten times across an entire piece.",
     "The ecosystem needs ecosystems to build ecosystem value."),
    ("Historical Analogy Stacking", "composition",
     "Rapid-fire listing of historical companies or examples to manufacture false authority.",
     "Apple didn't build Uber. Facebook didn't build Spotify."),
    ("One-Point Dilution", "composition",
     "A single argument restated ten different ways across thousands of words.",
     "The same point, restated eight ways across 4000 words."),
    ("Content Duplication", "composition",
     "Repeating an entire section verbatim within the same piece.",
     "The same section appeared twice, word-for-word identical."),
    ("The Signposted Conclusion", "composition",
     "Explicitly announcing a conclusion with a formula phrase instead of just concluding.",
     "In conclusion, the future of AI depends on..."),
    ("Despite Its Challenges", "composition",
     "A rigid formula that acknowledges a problem only to immediately dismiss it.",
     "Despite these challenges, the initiative continues to thrive."),
]


def main():
    ids = [str(uuid.uuid5(NAMESPACE, name)) for name, *_ in TROPES]
    names = [t[0] for t in TROPES]
    categories = [t[1] for t in TROPES]
    descriptions = [t[2] for t in TROPES]
    examples = [t[3] for t in TROPES]

    table = pa.table({
        "trope_id": pa.array(ids, type=pa.string()),
        "name": pa.array(names, type=pa.string()),
        "category": pa.array(categories, type=pa.string()),
        "description": pa.array(descriptions, type=pa.string()),
        "example_phrase": pa.array(examples, type=pa.string()),
    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pq.write_table(table, OUT, compression="zstd")
    print(f"wrote {len(TROPES)} tropes -> {OUT}")


if __name__ == "__main__":
    main()
