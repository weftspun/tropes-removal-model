# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
ONNX Runtime session helpers shared by gate.py. Auto-selects the fastest
available execution provider: CoreML (Metal/ANE) on macOS, else CPU.

Never silently falls back without logging -- `session_for` prints which
provider onnxruntime actually picked, so "why is this slow on my Mac" has an
immediate answer instead of a silent CPU fallback.
"""
import sys


def preferred_providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    order = []
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        order.append("CoreMLExecutionProvider")
    order.append("CPUExecutionProvider")
    return order


def session_for(onnx_path, log=True, register_custom_ops=None):
    """register_custom_ops: path to a custom-op shared library (e.g.
    onnxruntime_extensions.get_library_path()) -- needed for merged_model.onnx,
    whose tokenizer branch uses onnxruntime_extensions' ops."""
    import onnxruntime as ort
    providers = preferred_providers()
    so = ort.SessionOptions()
    if register_custom_ops:
        so.register_custom_ops_library(register_custom_ops)
    sess = ort.InferenceSession(onnx_path, so, providers=providers)
    if log:
        print(f"[infer] {onnx_path}: requested {providers} -> using {sess.get_providers()[0]}",
              file=sys.stderr)
    return sess
