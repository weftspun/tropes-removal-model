# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Compiles scripts/seed_labels.py's regex patterns into a real ONNX subgraph,
using the standard `RegexFullMatch` op (ai.onnx opset 20+, RE2 syntax) --
no custom ops, no onnxruntime-extensions dependency for this branch.
`RegexFullMatch` requires the *whole* string to match, so every pattern is
wrapped as `(?i)?.*<pattern>.*` to recover Python re.search()'s
"match anywhere in the string" semantics (verified: matches re.search's
case-(in)sensitive substring-match results on the same test sentences).

Deliberately reuses scripts.seed_labels.PATTERNS as the single source of
truth -- those regexes serve double duty: weak-label bootstrapping for
training data (scripts/seed_labels.py) AND the actual runtime detector for
the ~21 "mechanical" tropes (this module), per the decision to go hybrid:
deterministic detectors for lexical/structural/formatting tropes that don't
need a training example to detect correctly, a learned classifier only for
the ~12 tropes that are genuinely a judgment call (see export_onnx_tropes.py
for how this subgraph merges with the classifier's ONNX graph).
"""
import re
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, __file__.rsplit("runtime", 1)[0])
from scripts.seed_labels import PATTERNS

MECHANICAL_TROPE_NAMES = list(PATTERNS.keys())


def _to_re2_full_match(compiled_pattern):
    """Convert a Python `re` pattern (used with .search()) to an RE2 pattern
    usable with ONNX's RegexFullMatch (which anchors both ends).

    Three adjustments beyond just adding .* on both sides:
    - Wrap the original pattern in a non-capturing group first -- `.*A|B.*`
      parses as `(.*A)|(B.*)`, not `.*(A|B).*`, for any pattern with a
      top-level alternation (e.g. Anaphora Abuse's unrolled `x.*x|y.*y`).
    - Make ONLY the outer `.*` wrapper dotall (`(?s:.*)`), not the whole
      pattern: RE2's `.` doesn't match `\\n` by default, so the wrapper's
      `.*` can't span a newline unless told to. But the *original* pattern
      must keep Python's default (non-dotall) `.` semantics -- e.g. Tricolon
      Abuse's own internal `.*` between two `word;word` matches is meant to
      require them on the same line, same as Python's `re.search` (compiled
      with just re.I, no re.DOTALL) does. Making dotall global instead of
      scoped to just the outer wrapper broke that: confirmed by a
      20k-sentence audit finding a poem where two `word;word` occurrences
      separated by many newlines wrongly matched under a global `(?s)` but
      correctly didn't match either engine once scoped.
    - `(?-s:...)` around the original pattern makes this scoping explicit
      even though it's already the default, so this stays correct if a
      future pattern is written assuming its own `.` is non-dotall.
    Confirmed by the same 20k-sentence audit: 0 mismatches vs Python's
    re.search across all 21 mechanical tropes after this fix."""
    inline_flags = "(?i)" if compiled_pattern.flags & re.IGNORECASE else ""
    return f"{inline_flags}(?s:.*)(?-s:{compiled_pattern.pattern})(?s:.*)"


def build_regex_graph(trope_order):
    """Build an ONNX graph: input `text` (string[batch]) -> output
    `trope_scores` (float32[batch, len(trope_order)]), one column per name
    in `trope_order` that has a PATTERNS entry (1.0/0.0 match, deterministic
    -- not a probability). Names without a pattern get a constant 0 column,
    so the caller can pass the full 33-name canonical order and the merge
    step in export_onnx_tropes.py only needs to overlay the classifier's
    columns for the non-mechanical names."""
    nodes, cast_outputs = [], []
    for i, name in enumerate(trope_order):
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
    # opset>=13 moved Unsqueeze's axes from an attribute to a second input tensor
    axes_init = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1])
    unsqueeze_nodes = [
        helper.make_node("Unsqueeze", [c, "unsqueeze_axes"], [f"{c}_u"], name=f"unsq_{i}")
        for i, c in enumerate(cast_outputs)
    ]
    concat_node = helper.make_node(
        "Concat", [f"{c}_u" for c in cast_outputs], ["trope_scores"], axis=1, name="concat_scores")

    graph = helper.make_graph(
        [shape_node] + nodes + unsqueeze_nodes + [concat_node],
        "regex_tropes",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
        [helper.make_tensor_value_info("trope_scores", TensorProto.FLOAT, [None, len(trope_order)])],
        initializer=[axes_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model
