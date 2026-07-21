# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Generate synthetic (trope-positive, clean-rewrite) pairs per trope, plus
matched clean negatives, using an LLM seeded with each trope's tropes.fyi
description + example phrase. This is the primary fix for class imbalance
on the composition-level tropes (Fractal Summaries, Content Duplication,
One-Point Dilution, etc.) that public corpora rarely contain in isolation,
and the ONLY source of rewrite supervision for train_rewriter.py.

Writes:
  - source_document(is_synthetic=true) + sentence for each positive example
  - sentence_trope_label(label_source="synthetic-gen", confidence=1.0)
  - sentence_rewrite(source="synthetic-gen") linking the positive sentence
    to its trope-removed rewrite
  - source_document + sentence (no label, no rewrite) for clean negatives

Requires ANTHROPIC_API_KEY (or swap `LLMClient` below for another provider).
The client is intentionally a single narrow seam so the generation prompts/
schema stay stable if the backing model changes.
"""
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from runtime.datalake import (
    SENTENCE_PATH, SENTENCE_REWRITE_PATH, SENTENCE_TROPE_LABEL_PATH,
    SOURCE_DOCUMENT_PATH, append_rows, read_tropes,
)
from runtime.sentence_split import split_sentences

N_POSITIVE_PER_TROPE = 40
N_NEGATIVE_TOTAL = 400

DOC_NAMESPACE = uuid.UUID("9c4d9f5e-9f0a-4c2e-8e2f-2c9b5b7b6a55")
SENT_NAMESPACE = uuid.UUID("ad5eaf6e-af0a-4c2e-8e2f-2c9b5b7b6a66")
REWRITE_NAMESPACE = uuid.UUID("be6fbf7e-bf0a-4c2e-8e2f-2c9b5b7b6a77")


class LLMClient:
    """Thin seam over the generation backend. Default: Anthropic Messages API."""

    def __init__(self, model="claude-sonnet-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_json(self, prompt, max_tokens=4096):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        start, end = text.find("["), text.rfind("]")
        return json.loads(text[start:end + 1])


def _insert_document(text, is_synthetic, is_ai_generated, doc_rows, sent_rows):
    text = text.strip()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc_id = str(uuid.uuid5(DOC_NAMESPACE, sha))
    doc_rows.append({
        "doc_id": doc_id, "source": "synthetic", "url": None, "sha256": sha,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "is_ai_generated": is_ai_generated, "is_synthetic": is_synthetic,
    })
    sent_ids = []
    for i, sent in enumerate(split_sentences(text)):
        sent_id = str(uuid.uuid5(SENT_NAMESPACE, f"{doc_id}:{i}"))
        sent_rows.append({
            "sentence_id": sent_id, "doc_id": doc_id, "ordinal": i,
            "char_start": sent.char_start, "char_end": sent.char_end, "text": sent.text,
        })
        sent_ids.append(sent_id)
    return sent_ids


def generate_for_trope(client, trope_name, description, example, n):
    prompt = f"""You are building a training set for a classifier that detects AI-writing
"tropes" (recurring stylistic tells of AI-generated prose).

Trope: {trope_name}
Description: {description}
Example: {example!r}

Generate {n} short (1-3 sentence) examples that clearly exhibit this trope,
each paired with a clean rewrite that says the same thing WITHOUT the trope.
Vary topic and phrasing widely; do not just repeat the example.

Respond with ONLY a JSON array, no prose, no markdown fence:
[{{"positive": "...", "rewrite": "..."}}, ...]"""
    return client.generate_json(prompt)


def generate_negatives(client, n):
    prompt = f"""Generate {n} short (1-3 sentence) snippets of plain, clean, direct
human writing on varied everyday topics -- no AI-writing stylistic tells
(no em-dash overuse, no "delve", no rhetorical-question-then-answer,
no grandiose framing, no filler transitions). Just ordinary, specific prose.

Respond with ONLY a JSON array of strings, no prose, no markdown fence:
["...", "...", ...]"""
    return client.generate_json(prompt)


def main():
    client = LLMClient()
    tropes = read_tropes()
    doc_rows, sent_rows, label_rows, rewrite_rows = [], [], [], []

    for trope_id, name, description, example in zip(
        tropes.column("trope_id").to_pylist(), tropes.column("name").to_pylist(),
        tropes.column("description").to_pylist(), tropes.column("example_phrase").to_pylist(),
    ):
        print(f"generating for {name}...", file=sys.stderr)
        pairs = generate_for_trope(client, name, description, example, N_POSITIVE_PER_TROPE)
        for pair in pairs:
            pos_ids = _insert_document(pair["positive"], True, True, doc_rows, sent_rows)
            rewrite_ids = _insert_document(pair["rewrite"], True, False, doc_rows, sent_rows)
            for sid in pos_ids:
                label_rows.append({
                    "sentence_id": sid, "trope_id": trope_id,
                    "label_source": "synthetic-gen", "confidence": 1.0,
                })
            if pos_ids and rewrite_ids:
                rewrite_text = pair["rewrite"].strip()
                rewrite_rows.append({
                    "rewrite_id": str(uuid.uuid5(REWRITE_NAMESPACE, pos_ids[0])),
                    "sentence_id": pos_ids[0], "trope_id": trope_id,
                    "rewritten_text": rewrite_text, "source": "synthetic-gen",
                })

    print("generating clean negatives...", file=sys.stderr)
    negatives = generate_negatives(client, N_NEGATIVE_TOTAL)
    for neg in negatives:
        _insert_document(neg, True, False, doc_rows, sent_rows)

    append_rows(SOURCE_DOCUMENT_PATH, doc_rows)
    append_rows(SENTENCE_PATH, sent_rows)
    append_rows(SENTENCE_TROPE_LABEL_PATH, label_rows)
    append_rows(SENTENCE_REWRITE_PATH, rewrite_rows)
    print(f"synthetic: {len(doc_rows)} docs, {len(label_rows)} labels, "
          f"{len(rewrite_rows)} rewrite pairs", file=sys.stderr)


if __name__ == "__main__":
    main()
