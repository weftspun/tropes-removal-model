# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
The pre-commit/CI gate entry point.

    python gate.py <files...>              # pre-commit passes staged files as argv
    python gate.py --base-ref origin/main   # CI mode: diff against a base ref

For every sentence in every target file, runs ONE merged ONNX model
(onnx_tropes/merged_model.onnx -- see export_onnx_tropes.py) that scores all
33 tropes in a single call: a deterministic regex branch for the ~21
mechanical tropes, a distilbert classifier branch for the ~12 semantic ones.
Any trope above --threshold becomes a Finding with an exact file/line/char
span, the matching trope's name + category + description, and a suggested
rewrite from the ONNX rewriter. Never emits a bare document-level score --
every flag is pinned to one sentence and one named trope.

Exits non-zero (fails the gate) if total findings exceed --max-findings.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.datalake import read_tropes
from runtime.infer import session_for
from runtime.sentence_split import split_sentences

MERGED_MODEL_PATH = os.path.join("onnx_tropes", "merged_model.onnx")
ONNX_REWRITER_DIR = "onnx_rewriter"
TEXT_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}


class Finding:
    def __init__(self, file, line, char_start, char_end, sentence_text,
                 trope_name, category, confidence, description, suggested_rewrite):
        self.file = file
        self.line = line
        self.char_start = char_start
        self.char_end = char_end
        self.sentence_text = sentence_text
        self.trope_name = trope_name
        self.category = category
        self.confidence = confidence
        self.description = description
        self.suggested_rewrite = suggested_rewrite

    def report_line(self):
        return (f"{self.file}:{self.line} — [{self.trope_name}] ({self.confidence:.2f}) — "
                f"\"{self.sentence_text}\" — {self.description} — "
                f"suggested: \"{self.suggested_rewrite}\"")


class TropeClassifier:
    """Wraps the single merged ONNX model (regex branch + semantic
    classifier branch, see export_onnx_tropes.py) -- one session, one
    forward pass per sentence, one 33-long score vector back."""

    def __init__(self):
        tropes = read_tropes().to_pandas()
        self.names = tropes["name"].tolist()
        self.category_of = dict(zip(tropes["name"], tropes["category"]))
        self.description_of = dict(zip(tropes["name"], tropes["description"]))
        self.available = os.path.isfile(MERGED_MODEL_PATH)
        if self.available:
            import onnxruntime_extensions as ortx
            self.session = session_for(MERGED_MODEL_PATH, log=False, register_custom_ops=ortx.get_library_path())
        else:
            print(f"[gate] {MERGED_MODEL_PATH} not found -- no findings will fire until it's built "
                  "(see export_onnx_tropes.py)", file=sys.stderr)

    def score_all(self, text):
        """Returns {trope_name: confidence} for every trope in one call."""
        if not self.available:
            return {name: 0.0 for name in self.names}
        out = self.session.run(None, {"text": np.array([text], dtype=object)})[0]
        return dict(zip(self.names, out[0].tolist()))


class Rewriter:
    def __init__(self):
        self.available = os.path.isdir(ONNX_REWRITER_DIR)
        if self.available:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(ONNX_REWRITER_DIR)
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                ONNX_REWRITER_DIR, provider="CPUExecutionProvider")

    def rewrite(self, trope_name, sentence_text):
        if not self.available:
            return "(rewriter model unavailable)"
        prompt = f"remove {trope_name.lower()}: {sentence_text}"
        enc = self.tokenizer(prompt, return_tensors="pt")
        ids = self.model.generate(**enc, max_new_tokens=64, num_beams=1)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


def changed_files(base_ref):
    out = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACM", base_ref, "HEAD"],
                          capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f.strip()]


def is_text_file(path):
    return os.path.splitext(path)[1].lower() in TEXT_EXTENSIONS


def line_of(char_offset, text):
    return text.count("\n", 0, char_offset) + 1


def scan_file(path, classifier, rewriter, threshold):
    findings = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    for sentence in split_sentences(text):
        scores = classifier.score_all(sentence.text)
        for trope_name, conf in scores.items():
            if conf >= threshold:
                findings.append(Finding(
                    file=path, line=line_of(sentence.char_start, text),
                    char_start=sentence.char_start, char_end=sentence.char_end,
                    sentence_text=sentence.text, trope_name=trope_name,
                    category=classifier.category_of[trope_name],
                    confidence=conf, description=classifier.description_of[trope_name],
                    suggested_rewrite=rewriter.rewrite(trope_name, sentence.text),
                ))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="files to scan (pre-commit passes staged files)")
    parser.add_argument("--base-ref", help="CI mode: diff this ref against HEAD instead of using `files`")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-findings", type=int, default=0, help="0 = fail on any finding")
    args = parser.parse_args()

    targets = args.files
    if args.base_ref:
        targets = changed_files(args.base_ref)
    targets = [f for f in targets if is_text_file(f) and os.path.isfile(f)]

    if not targets:
        print("no text files to scan")
        return 0

    classifier = TropeClassifier()
    rewriter = Rewriter()

    all_findings = []
    for path in targets:
        all_findings.extend(scan_file(path, classifier, rewriter, args.threshold))

    for f in all_findings:
        print(f.report_line())

    print(f"\n{len(all_findings)} trope finding(s) across {len(targets)} file(s)")
    if len(all_findings) > args.max_findings:
        print(f"FAILED: exceeds --max-findings={args.max_findings}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
