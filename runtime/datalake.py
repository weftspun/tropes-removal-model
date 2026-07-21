# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
ETNF datalake tables, shared schema + read/append helpers.

Every table's primary key is a UUID string (never an integer surrogate).
Each satellite table is a binary projection on its own key with a foreign
key into `sentence`/`trope`, so its join dependency is implied by a
candidate key -- Essential Tuple Normal Form, matching the reasoning in the
fire/jobs-lazy-onboarding reference repo's build_db.py. Stored as
zstd-compressed parquet only; no JSON anywhere in the lake.

Tables that carry a trope_id/trope_name column are stored as Hive-partitioned
directories (one small parquet file per trope) rather than one monolithic
file -- see PARTITION_COLUMNS. `sentence`/`source_document` are NOT
partitioned this way: they have no single trope association (a sentence may
match zero, one, or several tropes), so partitioning them would mean copying
the full sentence text into 33 different partitions -- 33x the disk for
data that isn't trope-specific. Partitioning only pays off where a row
already belongs to exactly one trope.
"""
import os

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEEDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seeds")

TROPE_PATH = os.path.join(SEEDS_DIR, "trope.parquet")
SOURCE_DOCUMENT_PATH = os.path.join(DATA_DIR, "source_document.parquet")
SENTENCE_PATH = os.path.join(DATA_DIR, "sentence.parquet")
SENTENCE_TROPE_LABEL_PATH = os.path.join(DATA_DIR, "sentence_trope_label")
SENTENCE_REWRITE_PATH = os.path.join(DATA_DIR, "sentence_rewrite")
# ML-ready, derived from sentence_rewrite (see scripts/build_datalake.py) --
# small (source_text, trope_name, target_text) rows, one trope per row
# already, so partitioning is free reorganization, not duplication.
REWRITE_PAIRS_PATH = os.path.join(DATA_DIR, "rewrite_pairs")

# path -> partition column, for tables stored as Hive-partitioned directories
# (one parquet file per distinct value) instead of a single file.
PARTITION_COLUMNS = {
    SENTENCE_TROPE_LABEL_PATH: "trope_id",
    SENTENCE_REWRITE_PATH: "trope_id",
    REWRITE_PAIRS_PATH: "trope_name",
}

SCHEMAS = {
    SOURCE_DOCUMENT_PATH: pa.schema([
        ("doc_id", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
        ("sha256", pa.string()),
        ("fetched_at", pa.string()),
        ("is_ai_generated", pa.bool_()),
        ("is_synthetic", pa.bool_()),
    ]),
    SENTENCE_PATH: pa.schema([
        ("sentence_id", pa.string()),
        ("doc_id", pa.string()),
        ("ordinal", pa.int32()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("text", pa.string()),
    ]),
    SENTENCE_TROPE_LABEL_PATH: pa.schema([
        ("sentence_id", pa.string()),
        ("trope_id", pa.string()),
        ("label_source", pa.string()),
        ("confidence", pa.float32()),
    ]),
    SENTENCE_REWRITE_PATH: pa.schema([
        ("rewrite_id", pa.string()),
        ("sentence_id", pa.string()),
        ("trope_id", pa.string()),
        ("rewritten_text", pa.string()),
        ("source", pa.string()),
    ]),
    REWRITE_PAIRS_PATH: pa.schema([
        ("source_text", pa.string()),
        ("trope_name", pa.string()),
        ("target_text", pa.string()),
    ]),
}


def _write_table(path, table, schema):
    """Write `table` to `path` -- a single zstd parquet file, or (if `path`
    is a partitioned table) a Hive-partitioned directory of small zstd
    parquet files, one per distinct value of the partition column."""
    part_col = PARTITION_COLUMNS.get(path)
    if part_col is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        return
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    ds.write_dataset(
        table, path, format="parquet",
        partitioning=ds.partitioning(pa.schema([(part_col, pa.string())]), flavor="hive"),
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        existing_data_behavior="overwrite_or_ignore",
    )


def read_table(path):
    schema = SCHEMAS[path]
    if not os.path.exists(path):
        return pa.table({name: [] for name in schema.names}, schema=schema)
    if path in PARTITION_COLUMNS:
        part_col = PARTITION_COLUMNS[path]
        has_data = any(f.endswith(".parquet") for _, _, files in os.walk(path) for f in files)
        if not has_data:
            return pa.table({name: [] for name in schema.names}, schema=schema)
        table = ds.dataset(
            path, format="parquet",
            partitioning=ds.partitioning(pa.schema([(part_col, pa.string())]), flavor="hive"),
        ).to_table()
        return table.select(schema.names)
    return pq.read_table(path)


def append_rows(path, rows):
    """Append a list of dict rows to a table, rewriting it in place.
    Batch/offline use only (datalake build scripts), not a hot path.

    Drops exact full-row duplicates after concatenating, so re-running a
    build script against a persistent local data/ directory (e.g. adding
    more pairs to scripts/synth_seed_examples.py and re-running it) is
    idempotent instead of silently duplicating every previously-inserted
    row. All the *_id columns here are content-derived (uuid5 of a sha256),
    so a true re-insert is always byte-identical to the existing row."""
    if not rows:
        return
    schema = SCHEMAS[path]
    existing = read_table(path)
    new = pa.table({name: [r.get(name) for r in rows] for name in schema.names}, schema=schema)
    combined = pa.concat_tables([existing, new]).combine_chunks()
    combined = combined.to_pandas().drop_duplicates().reset_index(drop=True)
    combined = pa.Table.from_pandas(combined, schema=schema, preserve_index=False)
    _write_table(path, combined, schema)


def replace_rows(path, drop_column, drop_value, new_rows):
    """Drop every existing row where `drop_column == drop_value`, then append
    `new_rows`. For tables a script fully regenerates from a deterministic
    rule each run (e.g. seed_labels.py's regex pass) rather than growing
    incrementally -- re-running after editing the rule must replace stale
    output, not just add to it."""
    schema = SCHEMAS[path]
    existing = read_table(path).to_pandas()
    kept = existing[existing[drop_column] != drop_value]
    _write_table(path, pa.Table.from_pandas(kept, schema=schema, preserve_index=False), schema)
    append_rows(path, new_rows)


def read_tropes():
    if not os.path.exists(TROPE_PATH):
        raise FileNotFoundError(f"{TROPE_PATH} not found -- run scripts/seed_tropes.py first")
    return pq.read_table(TROPE_PATH)
