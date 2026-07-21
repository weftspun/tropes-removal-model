# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Fine-tune google/flan-t5-small on data/rewrite_pairs/ (Hive-partitioned by
trope_name) so one model
serves all 33 tropes' rewrites, conditioned on a trope-name prefix (e.g.
"remove em-dash addiction: <sentence>"). Plain HuggingFace Seq2SeqTrainer,
not AutoGluon -- AutoMM targets classification/regression/NER, not
open-ended generation, so this stage deliberately sits outside it. Being
plain torch (not AutoGluon) also means this script, unlike train_tropes.py,
actually gets Metal acceleration on a macOS runner: HF's Trainer picks up
torch's MPS backend automatically when torch.backends.mps.is_available(),
whereas AutoGluon's own docs say GPU/MPS isn't supported on macOS at all.

TRAIN_BATCH_SIZE (env, default 16) lets the CI caller cap memory: the free
macos-14 hosted runner has only 7GB RAM, so train-release.yml's
train-rewriter-metal job passes 8 to keep flan-t5-small's activations +
AdamW optimizer state comfortably under that ceiling; the Ubuntu path (if
ever run standalone) can use the default.

Output: models/rewriter/ (HF-format seq2seq checkpoint + tokenizer).
"""
import os
import warnings

import numpy as np
import pandas as pd
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKBONE = "google/flan-t5-small"
OUT_DIR = "models/rewriter"
MAX_LEN = 96
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))


def prefix(trope_name):
    return f"remove {trope_name.lower()}: "


def main():
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train_rewriter] device={device}  batch_size={BATCH_SIZE}", flush=True)

    df = pd.read_parquet("data/rewrite_pairs")
    df = df.dropna(subset=["source_text", "target_text", "trope_name"])
    df["input_text"] = df.apply(lambda r: prefix(r["trope_name"]) + r["source_text"], axis=1)

    n_val = max(1, int(len(df) * 0.1))
    val_df = df.sample(n=n_val, random_state=0)
    train_df = df.drop(val_df.index)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE)

    def tokenize(batch):
        model_in = tok(batch["input_text"], max_length=MAX_LEN, truncation=True)
        labels = tok(text_target=batch["target_text"], max_length=MAX_LEN, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_ds = Dataset.from_pandas(train_df[["input_text", "target_text"]], preserve_index=False).map(
        tokenize, batched=True, remove_columns=["input_text", "target_text"])
    val_ds = Dataset.from_pandas(val_df[["input_text", "target_text"]], preserve_index=False).map(
        tokenize, batched=True, remove_columns=["input_text", "target_text"])

    collator = DataCollatorForSeq2Seq(tok, model=model)
    # No explicit `use_mps_device` kwarg: recent transformers versions removed
    # it and auto-detect torch.backends.mps.is_available() on their own; the
    # `device` variable above is only for the log line and BATCH_SIZE sizing.
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=6, learning_rate=3e-4, predict_with_generate=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, logging_steps=20, report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved rewriter -> {OUT_DIR}")


if __name__ == "__main__":
    main()
