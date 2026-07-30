# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""Compiles scripts/seed_labels.py's regex patterns into a real ONNX subgraph,
using the standard `RegexFullMatch` op (ai.onnx opset 20+, RE2 syntax) --
no custom ops, no onnxruntime-extensions dependency for this branch.
`RegexFullMatch` requires the *whole* string to match, so every pattern is
wrapped as `(?i)?.*<pattern>.*` to recover Python re.search()'s
"match anywhere in the string" semantics.

Detects ~18 mechanical STE violations (passive voice, semicolons, contractions,
nominalizations, marketing adjectives, phrasal verbs, banned synonyms, modal
hedges, "-ing" main verbs, stacked auxiliaries, missing articles, em-dashes,
three+ nouns in a row, etc.) deterministically -- zero training data needed.

For the full ASD-STE100 specification, see https://asd-ste100.org.
"""
import re
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, __file__.rsplit("runtime", 1)[0])
from scripts.seed_labels import PATTERNS

MECHANICAL_VIOLATION_NAMES = list(PATTERNS.keys())


def _to_re2_full_match(compiled_pattern):
    """Convert a Python `re` pattern (used with .search()) to an RE2 pattern
    usable with ONNX's RegexFullMatch (which anchors both ends).

    Three adjustments beyond just adding .* on both sides:
    - Wrap the original pattern in a non-capturing group first -- `.*A|B.*`
      parses as `(.*A)|(B.*)`, not `.*(A|B).*`, for any pattern with a
      top-level alternation.
    - Make ONLY the outer `.*` wrapper dotall (`(?s:.*)`), not the whole
      pattern: RE2's `.` doesn't match `\n` by default; the original pattern
      must keep non-dotall `.` semantics (same as Python's `re.search` without
      re.DOTALL).
    - `(?-s:...)` around the original pattern makes the scoping explicit.
    """
    inline_flags = "(?i)" if compiled_pattern.flags & re.IGNORECASE else ""
    return f"{inline_flags}(?s:.*)(?-s:{compiled_pattern.pattern})(?s:.*)"


def build_regex_graph(violation_order):
    """Build an ONNX graph: input `text` (string[batch]) -> output
    `trope_scores` (float32[batch, len(violation_order)]), one column per name
    in `violation_order` that has a PATTERNS entry (1.0/0.0 match, deterministic
    -- not a probability). Names without a pattern get a constant 0 column,
    so the caller can pass the full canonical order and the merge step in
    export_onnx_tropes.py only needs to overlay the classifier's columns for
    the non-mechanical names."""
    nodes, cast_outputs = [], []
    for i, name in enumerate(violation_order):
        if name not in PATTERNS:
            nodes.append(helper.make_node(
                "ConstantOfShape", ["batch_shape"], [f"zero_{i}"],
                value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0]),
                name=f"zero_const_{i}"))
            cast_outputs.append(f"zero_{i}")
            continue
        pattern = _to_re2_full_match(PATTERNS[name])
        match_out, cast_out = f"match_{i}", f"cast_{i}"
        nodes.append(helper.make_node("RegexFullMatch", ["text"], [match_out],
                                       pattern=pattern, name=f"regex_{i}"))
        nodes.append(helper.make_node("Cast", [match_out], [cast_out],
                                       to=TensorProto.FLOAT, name=f"cast_{i}"))
        cast_outputs.append(cast_out)

    shape_node = helper.make_node("Shape", ["text"], ["batch_shape"], name="batch_shape_of_text")
    axes_init = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1])
    unsqueeze_nodes = [
        helper.make_node("Unsqueeze", [c, "unsqueeze_axes"], [f"{c}_u"], name=f"unsq_{i}")
        for i, c in enumerate(cast_outputs)
    ]
    concat_node = helper.make_node(
        "Concat", [f"{c}_u" for c in cast_outputs], ["trope_scores"], axis=1, name="concat_scores")

    graph = helper.make_graph(
        [shape_node] + nodes + unsqueeze_nodes + [concat_node],
        "regex_ste_violations",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
        [helper.make_tensor_value_info("trope_scores", TensorProto.FLOAT, [None, len(violation_order)])],
        initializer=[axes_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model