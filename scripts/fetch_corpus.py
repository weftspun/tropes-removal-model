# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Pull public AI-vs-human text corpora into the ETNF datalake's
source_document + sentence tables:

  - Hello-SimpleAI/HC3: short human/ChatGPT Q&A pairs.
  - artem9k/ai-text-detection-pile: long-form essays, human vs. GPT2/GPT3/
    ChatGPT/GPTJ. Long-form AI essays are far more likely to actually
    contain the composition-level tropes (Fractal Summaries, Signposted
    Conclusions, Historical Analogy Stacking, One-Point Dilution) than
    HC3's short QA answers, which is exactly why scripts/seed_labels.py's
    regex pass found so few hits for those tropes against HC3 alone.
  - liamdugan/raid (RAID benchmark): 11 generators x 8 domains, unattacked
    rows only (attack == "none"). The 11.7GB train.csv is too large to
    download whole, so this pulls small byte-range samples from several
    offsets spread across the file rather than streaming/downloading it in
    full -- the file is grouped by domain/model, not shuffled, so sampling
    only the head would get one domain and miss the rest. Adds domains
    (poetry, wiki, books, reddit) none of the other two sources cover.

This only populates raw text + is_ai_generated; it does NOT assign trope
labels (see scripts/seed_labels.py) or generate rewrites (see
scripts/synth_generate.py). Re-running is idempotent: runtime.datalake.
append_rows drops exact-duplicate rows, and content-derived doc_id/
sentence_id mean a re-fetched document lands on the same rows either way.

Requires network access to the Hugging Face Hub. No Kaggle source is
wired in: the larger Kaggle corpora (DAIGT-v2, Human vs. LLM Text Corpus,
AIDE) would add real volume, but the Kaggle API needs credentials
(KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json) that aren't present
in this environment -- add a kaggle.api pull here if that changes.
"""
import hashlib
import io
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
import requests
from datasets import load_dataset

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from runtime.datalake import SOURCE_DOCUMENT_PATH, SENTENCE_PATH, append_rows, read_table
from runtime.sentence_split import split_sentences

# artem9k/ai-text-detection-pile ships 7 parquet shards; source is not
# interleaved within a shard (confirmed by sampling: shard 0 is ~all
# "human", shard 6 is ~all "ai"), so pull one of each rather than the
# whole multi-hundred-thousand-row pile.
PILE_HUMAN_SHARD = "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00000-of-00007-bc5952582e004d67.parquet"
PILE_AI_SHARD = "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00006-of-00007-3d8a471ba0cf1c8d.parquet"
PILE_SAMPLE_PER_SHARD = 8000

RAID_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
RAID_HEADER = ["id", "adv_source_id", "source_id", "model", "decoding",
               "repetition_penalty", "attack", "domain", "title", "prompt", "generation"]
RAID_CHUNK_BYTES = 4_000_000
RAID_OFFSET_PCTS = (0, 15, 30, 45, 60, 75, 90)  # spread across the 11.7GB file; it's grouped by domain, not shuffled

DOC_NAMESPACE = uuid.UUID("6f1a6f2e-6d0a-4c2e-8e2f-2c9b5b7b6a22")
SENT_NAMESPACE = uuid.UUID("7a2b7f3e-7e0a-4c2e-8e2f-2c9b5b7b6a33")


def _existing_hashes():
    tbl = read_table(SOURCE_DOCUMENT_PATH)
    return set(tbl.column("sha256").to_pylist())


def ingest_text(text, source, url, is_ai_generated, seen_hashes, doc_rows, sent_rows):
    text = (text or "").strip()
    if not text:
        return
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if sha in seen_hashes:
        return
    seen_hashes.add(sha)
    doc_id = str(uuid.uuid5(DOC_NAMESPACE, sha))
    doc_rows.append({
        "doc_id": doc_id, "source": source, "url": url, "sha256": sha,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "is_ai_generated": is_ai_generated, "is_synthetic": False,
    })
    for i, sent in enumerate(split_sentences(text)):
        sent_id = str(uuid.uuid5(SENT_NAMESPACE, f"{doc_id}:{i}"))
        sent_rows.append({
            "sentence_id": sent_id, "doc_id": doc_id, "ordinal": i,
            "char_start": sent.char_start, "char_end": sent.char_end, "text": sent.text,
        })


def fetch_hc3(limit, seen, doc_rows, sent_rows):
    print("loading Hello-SimpleAI/HC3...", file=sys.stderr)
    # HC3's repo ships a legacy loading script the `datasets` lib no longer
    # executes; the HF-auto-converted parquet revision loads cleanly instead.
    ds = load_dataset("Hello-SimpleAI/HC3", split="train", revision="refs/convert/parquet")
    for row in ds.select(range(min(limit, len(ds)))):
        for human_answer in row.get("human_answers") or []:
            ingest_text(human_answer, "HC3", None, False, seen, doc_rows, sent_rows)
        for chatgpt_answer in row.get("chatgpt_answers") or []:
            ingest_text(chatgpt_answer, "HC3", None, True, seen, doc_rows, sent_rows)


def fetch_ai_text_detection_pile(seen, doc_rows, sent_rows):
    print("loading artem9k/ai-text-detection-pile (human + ai shards)...", file=sys.stderr)
    human_df = pd.read_parquet(PILE_HUMAN_SHARD, columns=["source", "text"])
    human_df = human_df[human_df["source"] == "human"].sample(
        n=min(PILE_SAMPLE_PER_SHARD, len(human_df)), random_state=0)
    ai_df = pd.read_parquet(PILE_AI_SHARD, columns=["source", "text"])
    ai_df = ai_df[ai_df["source"] != "human"].sample(
        n=min(PILE_SAMPLE_PER_SHARD, len(ai_df)), random_state=0)

    for text in human_df["text"]:
        ingest_text(text, "ai-text-detection-pile", None, False, seen, doc_rows, sent_rows)
    for text in ai_df["text"]:
        ingest_text(text, "ai-text-detection-pile", None, True, seen, doc_rows, sent_rows)


def fetch_raid(seen, doc_rows, sent_rows):
    print("loading liamdugan/raid (byte-range samples across the 11.7GB file)...", file=sys.stderr)
    head = requests.head(RAID_URL, allow_redirects=True, timeout=30)
    total = int(head.headers.get("Content-Length", 0))
    if not total:
        print("  could not determine RAID file size, skipping", file=sys.stderr)
        return

    frames = []
    for pct in RAID_OFFSET_PCTS:
        offset = total * pct // 100
        r = requests.get(RAID_URL, headers={"Range": f"bytes={offset}-{offset + RAID_CHUNK_BYTES}"},
                          timeout=60, allow_redirects=True)
        try:
            # Range cuts don't land on row boundaries; the python engine's
            # on_bad_lines='skip' tolerates the ragged first/last partial rows.
            df = pd.read_csv(io.BytesIO(r.content), names=RAID_HEADER, header=0, skiprows=1,
                              on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"  chunk at {pct}% failed to parse ({e}), skipping", file=sys.stderr)
            continue
        frames.append(df)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["attack"] == "none"]
    for row in combined.itertuples(index=False):
        ingest_text(row.generation, "RAID", None, row.model != "human", seen, doc_rows, sent_rows)


def main(hc3_limit=20000):
    seen = _existing_hashes()
    doc_rows, sent_rows = [], []

    fetch_hc3(hc3_limit, seen, doc_rows, sent_rows)
    fetch_ai_text_detection_pile(seen, doc_rows, sent_rows)
    fetch_raid(seen, doc_rows, sent_rows)

    append_rows(SOURCE_DOCUMENT_PATH, doc_rows)
    append_rows(SENTENCE_PATH, sent_rows)
    print(f"ingested {len(doc_rows)} documents, {len(sent_rows)} sentences", file=sys.stderr)


if __name__ == "__main__":
    main()
