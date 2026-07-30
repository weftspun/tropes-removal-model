# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
The pre-commit/CI gate entry point for ASD-STE100 compliance.

    python gate.py <files...>              # pre-commit passes staged files as argv
    python gate.py --base-ref origin/main   # CI mode: diff against a base ref

For every sentence in every target file, runs ONE merged ONNX model
(onnx_violations/merged_model.onnx -- see export_onnx_tropes.py) that scores
the STE violations in a single call: a deterministic regex branch for the ~18
mechanical violations, a SetFit classifier branch for the ~8 genuinely
semantic ones. The remaining 2 (Content Duplication, Historical Analogy
Stacking) are document-scoped-but-still-mechanical -- handled by a separate
deterministic whole-document pass, see runtime/cross_sentence.py.

Any violation above its threshold becomes a Finding with an exact file/line/char
span, the matching violation name + category + description, and a suggested
STE-compliant rewrite from the ONNX rewriter.

Exits non-zero (fails the gate) if total findings exceed --max-findings.

Not a certified ASD-STE100 checker. See https://asd-ste100.org for the spec.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.cross_sentence import CROSS_SENTENCE_VIOLATION_NAMES
from runtime.cross_sentence import detect as detect_cross_sentence_violations
from runtime.datalake import read_tropes
from runtime.infer import session_for
from runtime.sentence_split import split_sentences

MERGED_MODEL_PATH = os.path.join("onnx_violations", "merged_model.onnx")
THRESHOLDS_PATH = os.path.join("onnx_violations", "thresholds.json")
ONNX_REWRITER_DIR = "onnx_rewriter"
TEXT_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}

MAX_SCORED_CHARS = 1800


class Finding:
    def __init__(self, file, line, char_start, char_end, sentence_text,
                 violation_name, category, confidence, description, suggested_rewrite):
        self.file = file
        self.line = line
        self.char_start = char_start
        self.char_end = char_end
        self.sentence_text = sentence_text
        self.violation_name = violation_name
        self.category = category
        self.confidence = confidence
        self.description = description
        self.suggested_rewrite = suggested_rewrite

    def report_line(self):
        return (f"{self.file}:{self.line} — [{self.violation_name}] ({self.confidence:.2f}) — "
                f"\"{self.sentence_text}\" — {self.description} — "
                f"suggested: \"{self.suggested_rewrite}\"")


class ViolationClassifier:
    """Wraps the single merged ONNX model (regex branch + semantic
    classifier branch, see export_onnx_tropes.py) -- one session, one
    forward pass per sentence, one score vector back."""

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

        self.thresholds = {}
        if os.path.isfile(THRESHOLDS_PATH):
            import json
            with open(THRESHOLDS_PATH, encoding="utf-8") as fh:
                self.thresholds = json.load(fh)

    def score_all(self, text):
        """Returns {violation_name: confidence} for every violation in one call."""
        if not self.available:
            return {name: 0.0 for name in self.names}
        if len(text) > MAX_SCORED_CHARS:
            text = text[:MAX_SCORED_CHARS]
        out = self.session.run(None, {"text": np.array([text], dtype=object)})[0]
        return dict(zip(self.names, out[0].tolist()))

    def threshold_for(self, name, default):
        return self.thresholds.get(name, default)


class Rewriter:
    def __init__(self):
        self.available = os.path.isdir(ONNX_REWRITER_DIR)
        if self.available:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(ONNX_REWRITER_DIR)
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                ONNX_REWRITER_DIR, provider="CPUExecutionProvider", use_cache=False)

    def rewrite(self, violation_name, sentence_text, max_new_tokens=64):
        """Greedy-decodes by calling the model's forward() directly in a loop
        instead of HF's generate()."""
        import torch

        if not self.available:
            return "(rewriter model unavailable)"
        prompt = f"remove {violation_name.lower()}: {sentence_text}"
        enc = self.tokenizer(prompt, return_tensors="pt")
        decoder_start_id = self.model.config.decoder_start_token_id
        eos_id = self.model.config.eos_token_id

        decoder_input_ids = torch.tensor([[decoder_start_id]])
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    decoder_input_ids=decoder_input_ids, use_cache=False)
                next_id = out.logits[:, -1].argmax(dim=-1)
                if next_id.item() == eos_id:
                    break
                generated.append(next_id.item())
                decoder_input_ids = torch.cat([decoder_input_ids, next_id[:, None]], dim=-1)
        return self.tokenizer.decode(generated, skip_special_tokens=True)


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
    sentences = list(split_sentences(text))

    for sentence in sentences:
        scores = classifier.score_all(sentence.text)
        for violation_name, conf in scores.items():
            if violation_name in CROSS_SENTENCE_VIOLATION_NAMES:
                continue
            if conf >= classifier.threshold_for(violation_name, threshold):
                findings.append(Finding(
                    file=path, line=line_of(sentence.char_start, text),
                    char_start=sentence.char_start, char_end=sentence.char_end,
                    sentence_text=sentence.text, violation_name=violation_name,
                    category=classifier.category_of[violation_name],
                    confidence=conf, description=classifier.description_of[violation_name],
                    suggested_rewrite=rewriter.rewrite(violation_name, sentence.text),
                ))

    cross_sentence_hits = detect_cross_sentence_violations(text, sentences)
    for i, violation_names in cross_sentence_hits.items():
        sentence = sentences[i]
        for violation_name in violation_names:
            findings.append(Finding(
                file=path, line=line_of(sentence.char_start, text),
                char_start=sentence.char_start, char_end=sentence.char_end,
                sentence_text=sentence.text, violation_name=violation_name,
                category=classifier.category_of[violation_name],
                confidence=1.0, description=classifier.description_of[violation_name],
                suggested_rewrite=rewriter.rewrite(violation_name, sentence.text),
            ))
    return findings


def main():
    parser = argparse.ArgumentParser(description="ASD-STE100 compliance gate for technical prose")
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

    classifier = ViolationClassifier()
    rewriter = Rewriter()

    all_findings = []
    for path in targets:
        all_findings.extend(scan_file(path, classifier, rewriter, args.threshold))

    for f in all_findings:
        print(f.report_line())

    print(f"\n{len(all_findings)} STE violation finding(s) across {len(targets)} file(s)")
    if len(all_findings) > args.max_findings:
        print(f"FAILED: exceeds --max-findings={args.max_findings}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())