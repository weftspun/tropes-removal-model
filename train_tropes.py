# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Train the semantic-trope classifier with SetFit (contrastive
sentence-transformer fine-tuning + a lightweight multi-label head), not
AutoGluon's full end-to-end fine-tune -- and only for the ~8 genuinely
SEMANTIC tropes, not all 33.

Why SetFit over full fine-tuning: SetFit is purpose-built for exactly this
regime (few-shot text classification; the reference use case in its own
docs is 8-16 examples/class). Full fine-tuning of a 66M-param backbone
needs hundreds of examples per class to generalize instead of memorize;
SetFit's contrastive-pair pretraining step gets useful signal out of far
fewer labels because it's training a much smaller effective decision
surface (a frozen-ish sentence embedding + a linear/logistic head), not
updating every weight in a large encoder from scratch. Verified with a
real smoke test on this repo's own data (2 tropes, 648 rows, 1 epoch):
98.5% held-out accuracy and correct predictions on unseen sentences.

Why only ~8 tropes, not ~12: the ~23 mechanical tropes (Em-Dash Addiction,
Delve and Friends, Tricolon Abuse, etc.) already have a regex pattern in
scripts/seed_labels.py that detects them deterministically -- see
runtime/regex_onnx.py (verified 0 mismatches vs Python's re.search across
30,000 real sentences). Spending labeled data on those would be solving
something already solved exactly. Two more (Content Duplication, Historical
Analogy Stacking) turned out to be document-scoped-but-still-mechanical, not
genuinely fuzzy -- moved to a deterministic cross-sentence pass instead (see
runtime/cross_sentence.py) after two SetFit variants (single-sentence, then
a windowed-context retry) both failed to learn them reliably: a pooled
sentence embedding just isn't the right representation for "does this recur
elsewhere in the document." The rewriter (train_rewriter.py) is unaffected
by any of this -- it still needs rewrite pairs for all 33, since suggesting
a rewrite is a generation task regardless of how the trope was detected.

Output: models/setfit_classifier/ -- ONE multi-label SetFit model covering
the ~8 remaining semantic tropes (not one predictor per trope), since
SetFit's multi_target_strategy="one-vs-rest" natively handles co-occurring
labels with a single shared sentence-transformer body.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.cross_sentence import CROSS_SENTENCE_TROPE_NAMES
from runtime.regex_onnx import MECHANICAL_TROPE_NAMES

EXCLUDED_TROPE_NAMES = set(MECHANICAL_TROPE_NAMES) | set(CROSS_SENTENCE_TROPE_NAMES)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # same ONNX-export-reliable family as the reference repo
MODELS_DIR = "models/setfit_classifier"
NEGATIVE_TO_POSITIVE_RATIO = 3  # cap negatives so contrastive pairs aren't mostly negative-negative
NUM_EPOCHS = 1
NUM_ITERATIONS = 20  # contrastive pairs generated per training example


def main():
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train = pd.read_parquet("data/classifier_train.parquet")
    val = pd.read_parquet("data/classifier_val.parquet")
    all_label_cols = [c for c in train.columns if c.startswith("trope__")]
    label_cols = [c for c in all_label_cols if c[len("trope__"):] not in EXCLUDED_TROPE_NAMES]
    trope_names = [c[len("trope__"):] for c in label_cols]

    print(f"training ONE multi-label SetFit model for {len(label_cols)} semantic tropes "
          f"on backbone={BACKBONE} ({len(MECHANICAL_TROPE_NAMES)} mechanical tropes handled by "
          f"regex, see runtime/regex_onnx.py; {len(CROSS_SENTENCE_TROPE_NAMES)} document-scoped "
          "tropes handled deterministically, see runtime/cross_sentence.py)", flush=True)

    def build_dataset(df):
        pos_mask = df[label_cols].any(axis=1)
        positives = df[pos_mask]
        n_neg = min(len(df) - len(positives), len(positives) * NEGATIVE_TO_POSITIVE_RATIO)
        negatives = df[~pos_mask].sample(n=max(n_neg, 1), random_state=0)
        combined = pd.concat([positives, negatives]).sample(frac=1.0, random_state=0).reset_index(drop=True)
        labels = combined[label_cols].to_numpy(dtype=np.float32)
        return Dataset.from_dict({"text": combined["text"].tolist(), "label": labels.tolist()}), combined

    train_ds, train_df = build_dataset(train)
    val_ds, val_df = build_dataset(val)
    print(f"train: {len(train_ds)} rows ({int(train_df[label_cols].to_numpy().any(axis=1).sum())} positive)", flush=True)

    # use_differentiable_head (pure PyTorch head) instead of the default
    # sklearn LogisticRegression head: the sklearn head's skl2onnx export
    # produces float64 weights, which fails ONNX type-checking against the
    # sentence-transformer body's float32 output (MatMul type mismatch,
    # confirmed when exporting the sklearn-head version) -- an all-torch
    # model traces to ONNX as consistently float32 with no such conflict.
    model = SetFitModel.from_pretrained(
        BACKBONE, multi_target_strategy="one-vs-rest",
        use_differentiable_head=True, head_params={"out_features": len(label_cols)})
    args = TrainingArguments(
        batch_size=16, num_epochs=NUM_EPOCHS, num_iterations=NUM_ITERATIONS,
        output_dir=os.path.join(MODELS_DIR, "checkpoints"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"val metrics: {metrics}", flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_pretrained(MODELS_DIR)
    # trope name order must match the label vector's column order for
    # export_onnx_tropes.py to map ONNX output columns back to trope names
    with open(os.path.join(MODELS_DIR, "trope_order.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(trope_names))

    # Per-trope decision thresholds, not one global 0.5 for all 8: a single
    # multi-label head's per-class score distributions aren't calibrated
    # against each other (e.g. Short Punchy Fragments' real positives
    # clustered at 0.37-0.43 on this repo's data -- a real signal, just
    # sitting under a threshold tuned for other tropes). Picked by best F1
    # on val (not test, so gate.py's own eval stays honest), one scan per
    # trope over the same val predictions already computed by evaluate()
    # above -- cheap, no extra training.
    #
    # Unconstrained best-F1 search was verified to generalize on this repo's
    # own held-out test split, but a real-web audit against 100 genuinely
    # unseen AI-generated documents caught it overfitting anyway: Short
    # Punchy Fragments' precision estimate (0.67) came from only 10 val
    # positives -- too small a sample for the resulting low threshold (0.27)
    # to be trustworthy -- and it false-fired on ordinary, non-fragment
    # sentences in the wild ("There are many different search engines
    # available, such as Google, Bing, and Yahoo." is not a fragment).
    # MIN_PRECISION rules out thresholds whose val precision estimate is too
    # shaky to act on; among those that clear it, still prefer best F1
    # rather than max precision, so a trope doesn't get pinned needlessly
    # conservative when it has plenty of clean signal.
    MIN_PRECISION = 0.75
    val_probs = model.predict_proba(val_df["text"].tolist())
    val_probs = val_probs.detach().cpu().numpy() if hasattr(val_probs, "detach") else np.array(val_probs)
    val_labels = val_df[label_cols].to_numpy(dtype=np.float32)
    thresholds = {}
    for i, name in enumerate(trope_names):
        y, p = val_labels[:, i], val_probs[:, i]
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.1, 0.95, 0.01):
            pred = (p >= t).astype(np.float32)
            tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec < MIN_PRECISION:
                continue
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[name] = round(float(best_t), 2)
    import json
    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)
    print(f"per-trope thresholds: {thresholds}", flush=True)
    print(f"saved -> {MODELS_DIR}", flush=True)


if __name__ == "__main__":
    main()
