# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""Write seeds/trope.parquet: the ASD-STE100 violation categories
detected by this gate, normalized to the `trope` relation
(trope_id UUID, name, category, description, example_phrase).

These describe STE violations — patterns that violate the writing rules
and controlled dictionary of ASD-STE100 Issue 9. They are NOT an official
STE checker; see https://asd-ste100.org for the complete specification.

trope_id is a UUIDv5 derived from the violation name (fixed namespace),
re-running is idempotent. No JSON is written.
"""
import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

NAMESPACE = uuid.UUID("d9f6a9f2-9d0a-4c2e-8e2f-2c9b5b7b6a11")  # fixed; do not change
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seeds", "trope.parquet")

# (name, category, description, example_phrase)
# Mechanical violations — caught by deterministic regex in the ONNX graph.
# These correspond to ASD-STE100 writing rules.
VIOLATIONS = [
    ("Passive Voice", "verb-form",
     "STE Rule 1.5: Use active voice, not passive. Active voice tells the reader who does what.",
     "The file is read by the parser."),
    ("Sentence Too Long (procedural)", "sentence-structure",
     "STE Rule 3.3: Procedural sentences must not exceed 20 words. One instruction per sentence.",
     "Remove the four screws that hold the cover in place and then lift the cover carefully from the assembly."),
    ("Sentence Too Long (descriptive)", "sentence-structure",
     "STE Rule 3.3: Descriptive sentences must not exceed 25 words.",
     "The hydraulic system operates the landing gear, the flaps, the brakes, and the nose wheel steering mechanism simultaneously for improved ground handling."),
    ("Semicolon Used", "punctuation",
     "STE Rule 8.3: Use a period, not a semicolon. Write two separate sentences.",
     "Open the panel; remove the fuse."),
    ("Contraction Used", "word-form",
     "STE Rule 2.2: Do not use contractions. Write the full form.",
     "Don't remove the screws."),
    ("Nominalization", "verb-form",
     "STE Rule 1.10: Use a verb for an action, not a noun form. Write 'analyze' not 'perform an analysis'.",
     "The mechanic performs an analysis of the log."),
    ("Marketing Adjective", "word-choice",
     "STE approved dictionary: Use plain descriptive words, not empty superlatives or marketing jargon.",
     "Our robust, cutting-edge solution seamlessly integrates powerful features."),
    ("Phrasal Verb", "word-choice",
     "STE Rule 2.5: Use a single-word verb, not a phrasal verb with a preposition. Write 'start' not 'spin up'.",
     "Spin up the cluster before you dive into the configuration."),
    ("Banned Synonym", "word-choice",
     "STE approved dictionary: Use the short common word. 'start' not 'commence/initiate'; 'use' not 'utilize/leverage'; 'about' not 'regarding/concerning'; 'show' not 'demonstrate'; 'also' not 'additionally/furthermore'.",
     "We will utilize this framework to facilitate the process."),
    ("Modal Hedge", "sentence-structure",
     "STE Rule 3.5: Do not use empty filler phrases that add no instruction. State commands and facts directly.",
     "It is important to note that the system may fail if you do not follow these steps."),
    ("-ing Main Verb", "verb-form",
     "STE Rule 1.5: Prefer a simple tense over a progressive form for instructions.",
     "The system is showing the error message."),
    ("Stacked Auxiliaries", "verb-form",
     "STE Rule 1.5: Do not stack auxiliary verbs. State the action directly.",
     "This may help to improve the stability of the system."),
    ("Missing Article", "word-form",
     "STE Rule 2.1: Use an article (a, an, the, this, these) before a noun unless the noun is a technical name.",
     "Turn shaft assembly to access filter."),
    ("Em-Dash Overuse", "formatting",
     "STE Rule 8.1: Use standard punctuation. Em dashes for dramatic pauses are not technical writing.",
     "The problem — and this is the part nobody talks about — is scale."),
    ("Long Paragraph", "paragraph-structure",
     "STE Rule 4.1: One topic per paragraph, maximum six sentences.",
     "Paragraph with 7+ sentences covering multiple topics."),
    ("Unclear Instruction Order", "sentence-structure",
     "STE Rule 3.8: Put a condition before its command, not after.",
     "Remove the cover if the test fails."),
    ("Three+ Nouns in a Row", "word-form",
     "STE Rule 2.3: Do not use more than three nouns consecutively. Split the noun stack.",
     "overhead panel battery section connector"),
    ("Stacked Prepositional Phrases", "sentence-structure",
     "STE Rule 3.4: Do not use more than three prepositional phrases in one sentence.",
     "The data from the test of the system with the new software in the lab was sent."),
]

# Document-scoped mechanical violations — caught by cross_sentence.py
CROSS_SENTENCE_VIOLATIONS = [
    ("Content Duplication", "composition",
     "STE Rule: Do not repeat the same instruction or information verbatim in the same document.",
     "The same section appeared twice, word-for-word identical."),
    ("Historical Analogy Stacking", "composition",
     "STE Rule: Use relevant examples only. Rapid-fire listing of historical analogies is not concise technical writing.",
     "Apple didn't build Uber. Facebook didn't build Spotify."),
]

# Semantic violations — caught by the SetFit classifier (fuzzy judgment needed)
SEMANTIC_VIOLATIONS = [
    ("Word Choice Appropriateness", "word-choice",
     "Does the chosen word match the STE approved meaning? STE gives each approved word one meaning only.",
     "The pump 'falls' to deliver pressure."),
    ("Safety-Critical Clarity", "tone",
     "Does the warning or caution tell the reader what WILL happen, not what MAY happen? STE safety rules require definite language.",
     "The system may fail if you do not follow these steps."),
    ("Readability for Non-Native Speakers", "tone",
     "Would a reader with limited English understand this sentence on first reading? STE was designed for global readers.",
     "Subsequent to the cessation of the operational phase."),
    ("Technical Noun Consistency", "composition",
     "Do all technical names refer to the same item by the same name throughout? STE requires one name for one thing.",
     "The valve... later called the regulator... then the flow controller."),
    ("Sentence Logic (Makes Good Sense)", "composition",
     "Does the sentence hold together logically? STE Rule: every sentence must 'make good sense' — a judgment call.",
     "To prevent damage, press the button."),
    ("Appropriate Technical Verb", "word-choice",
     "Is the verb approved for this technical context? Technical verbs must match STE's approved list or be defined per project.",
     "The system 'facilitates' data transfer."),
    ("Unambiguous Reference", "tone",
     "STE requires each pronoun or reference to have exactly one clear antecedent. Ambiguous 'it', 'this', 'they' cause maintenance errors.",
     "Install the bracket and the plate. Then paint it."),
    ("Excessive Abstraction", "word-choice",
     "STE prefers concrete, specific words over abstract generalities. 'Turn the screw' not 'Manipulate the fastener'.",
     "Manipulate the fastening apparatus."),
]

ALL_VIOLATIONS = VIOLATIONS + CROSS_SENTENCE_VIOLATIONS + SEMANTIC_VIOLATIONS
CATEGORY_ORDER = [
    "verb-form", "sentence-structure", "punctuation", "word-form",
    "word-choice", "formatting", "paragraph-structure", "composition", "tone",
]


def main():
    ids = [str(uuid.uuid5(NAMESPACE, name)) for name, *_ in ALL_VIOLATIONS]
    names = [v[0] for v in ALL_VIOLATIONS]
    categories = [v[1] for v in ALL_VIOLATIONS]
    descriptions = [v[2] for v in ALL_VIOLATIONS]
    examples = [v[3] for v in ALL_VIOLATIONS]

    table = pa.table({
        "trope_id": pa.array(ids, type=pa.string()),
        "name": pa.array(names, type=pa.string()),
        "category": pa.array(categories, type=pa.string()),
        "description": pa.array(descriptions, type=pa.string()),
        "example_phrase": pa.array(examples, type=pa.string()),
    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pq.write_table(table, OUT, compression="zstd")
    print(f"wrote {len(ALL_VIOLATIONS)} STE violations -> {OUT}")


if __name__ == "__main__":
    main()