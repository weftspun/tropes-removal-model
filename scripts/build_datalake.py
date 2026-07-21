# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Join the normalized ETNF tables (sentence, sentence_trope_label, trope) into
a wide, denormalized train/val/test split for train_tropes.py: one row per
sentence, one boolean column per trope (`trope__<name>`), stratified by
trope x category so every trope has representation in each split.

This wide table is a derived ML-ready artifact, not part of the lake itself
-- the normalized tables in data/*.parquet remain the source of truth and
stay ETNF; this script only ever reads them.

Also emits data/rewrite_pairs/ (Hive-partitioned by trope_name -- sentence
text -> rewritten_text) for train_rewriter.py, built from sentence_rewrite.

classifier_{train,val,test}.parquet stay single files, NOT partitioned by
trope: every sentence is a potential negative example for all 33 tropes, so
splitting this table by trope would copy the full sentence text into 33
partitions instead of one -- 33x the disk for a table already close to the
per-file size ceiling. rewrite_pairs is small and each row already belongs
to exactly one trope, so partitioning it is free reorganization, not
duplication -- see runtime/datalake.py's module docstring.
"""
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from runtime.datalake import (
    DATA_DIR, REWRITE_PAIRS_PATH, SCHEMAS, SENTENCE_PATH, SENTENCE_REWRITE_PATH,
    SENTENCE_TROPE_LABEL_PATH, _write_table, read_table, read_tropes,
)

VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 0


def build_classifier_table():
    sentences = read_table(SENTENCE_PATH).to_pandas()
    labels = read_table(SENTENCE_TROPE_LABEL_PATH).to_pandas()
    tropes = read_tropes().to_pandas()

    id_to_name = dict(zip(tropes.trope_id, tropes.name))
    col_of = {tid: f"trope__{name}" for tid, name in id_to_name.items()}

    wide = sentences[["sentence_id", "text"]].drop_duplicates("sentence_id").set_index("sentence_id")
    for col in col_of.values():
        wide[col] = False

    # highest-confidence label per (sentence, trope) wins if duplicated across sources
    labels = labels.sort_values("confidence", ascending=False).drop_duplicates(["sentence_id", "trope_id"])
    for row in labels.itertuples(index=False):
        col = col_of.get(row.trope_id)
        if col is not None and row.sentence_id in wide.index:
            wide.loc[row.sentence_id, col] = row.confidence >= 0.5

    wide = wide.reset_index()
    return wide


def stratified_split(df, label_cols):
    # simplest stratification that keeps every trope represented in val/test:
    # split within each "primary trope" group (first true label, else "none")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    primary = df[label_cols].idxmax(axis=1)
    primary = primary.where(df[label_cols].any(axis=1), other="none")

    val_parts, test_parts, train_parts = [], [], []
    for _, group in df.groupby(primary):
        n = len(group)
        n_val = max(1, int(n * VAL_FRAC)) if n > 5 else 0
        n_test = max(1, int(n * TEST_FRAC)) if n > 5 else 0
        val_parts.append(group.iloc[:n_val])
        test_parts.append(group.iloc[n_val:n_val + n_test])
        train_parts.append(group.iloc[n_val + n_test:])
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(val_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))


def build_rewrite_table():
    rewrites = read_table(SENTENCE_REWRITE_PATH).to_pandas()
    sentences = read_table(SENTENCE_PATH).to_pandas().set_index("sentence_id")
    tropes = read_tropes().to_pandas().set_index("trope_id")

    rows = []
    for row in rewrites.itertuples(index=False):
        if row.sentence_id not in sentences.index or row.trope_id not in tropes.index:
            continue
        rows.append({
            "source_text": sentences.loc[row.sentence_id, "text"],
            "trope_name": tropes.loc[row.trope_id, "name"],
            "target_text": row.rewritten_text,
        })
    return pd.DataFrame(rows, columns=["source_text", "trope_name", "target_text"])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    wide = build_classifier_table()
    label_cols = [c for c in wide.columns if c.startswith("trope__")]
    train, val, test = stratified_split(wide, label_cols)
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(DATA_DIR, f"classifier_{name}.parquet")
        pq.write_table(pa.Table.from_pandas(split, preserve_index=False), path, compression="zstd")
        print(f"{name}: {len(split)} sentences -> {path}", file=sys.stderr)

    rewrite_df = build_rewrite_table()
    rewrite_table = pa.Table.from_pandas(rewrite_df, schema=SCHEMAS[REWRITE_PAIRS_PATH], preserve_index=False)
    _write_table(REWRITE_PAIRS_PATH, rewrite_table, SCHEMAS[REWRITE_PAIRS_PATH])
    print(f"rewrite pairs: {len(rewrite_df)} -> {REWRITE_PAIRS_PATH}/", file=sys.stderr)


if __name__ == "__main__":
    main()
