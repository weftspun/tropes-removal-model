# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Build the ONE merged ONNX model gate.py loads: single input `text`
(string[1] -- one sentence per call, matching how gate.py already invokes
it), single output `trope_scores` (float32[1, 33], canonical trope order
from seeds/trope.parquet).

Two branches, assembled into one graph:
  1. The regex branch (runtime/regex_onnx.py) -- ~21 mechanical tropes,
     deterministic RegexFullMatch nodes, zero training data. VERIFIED: 0
     mismatches vs Python's re.search across 30,000 real sentences. Its
     output is already a full 33-column vector with zeros in the semantic
     slots (see build_regex_graph's docstring).
  2. The semantic branch -- a tokenizer (onnxruntime_extensions,
     auto-generated from the SetFit body's own tokenizer) spliced onto the
     SetFit classifier exported via setfit.exporters.onnx.export_onnx().
     Its output is padded into the same 33-column shape, zeros in the
     mechanical slots.

Both branches consume the same `text` input and are additively combined
(`Add`): since each column is zero in exactly one branch, elementwise sum
reassembles the full 33-column vector without any dynamic gather/scatter.
`onnx.compose.add_prefix` avoids node/tensor name collisions between the
two independently-built subgraphs before they're spliced into one GraphProto.

Also exports the flan-t5-small rewriter separately (onnx_rewriter/) -- that
one stays a standalone seq2seq model; a generation model doesn't fold into
the same "one classification graph" simplification as the two branches
above, and optimum's seq2seq export already produces its own encoder/decoder
pair.
"""
import os
import warnings

import numpy as np
import onnx
from onnx import TensorProto, helper

warnings.filterwarnings("ignore")

ONNX_TROPES_DIR = "onnx_tropes"
ONNX_REWRITER_DIR = "onnx_rewriter"
MERGED_MODEL_PATH = os.path.join(ONNX_TROPES_DIR, "merged_model.onnx")
CLASSIFIER_MODEL_DIR = "models/setfit_classifier"
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # must match train_tropes.py's SetFit body


def _trope_order():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from runtime.datalake import read_tropes
    return read_tropes().to_pandas()["name"].tolist()


def _build_tokenizer_branch():
    """text (string[1]) -> input_ids, attention_mask (int64[1, seq_len]).
    onnxruntime_extensions' ragged/flattened per-batch-item output is a 1D
    [total_tokens] tensor (proven by direct test), not padded [batch,
    seq_len] -- fine here because gate.py only ever scores one sentence per
    call, so Unsqueeze adds the batch dim a single-sentence model needs."""
    import onnxruntime_extensions as ortx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    pre_model, _ = ortx.gen_processing_models(
        tok, pre_kwargs={"WITH_DEFAULT_INPUTS": True, "CAST_TOKEN_ID": True})

    # gen_processing_models' opset_import lists the standard domain as
    # 'ai.onnx' (not '', its usual alias) -- the checker wants an exact
    # match for the '' domain the Unsqueeze node below implicitly uses.
    onnx_domain_version = next(
        (op.version for op in pre_model.opset_import if op.domain in ("", "ai.onnx")), 18)
    pre_model.opset_import.append(helper.make_opsetid("", onnx_domain_version))

    axes_init = helper.make_tensor("tok_unsqueeze_axes", TensorProto.INT64, [1], [0])
    pre_model.graph.initializer.append(axes_init)
    # token_type_ids too -- the differentiable-head SetFit export (verified
    # via a real smoke test) takes 3 inputs (input_ids, attention_mask,
    # token_type_ids), not just 2; gen_processing_models already produces
    # token_type_ids as one of its outputs, this branch just wasn't keeping it.
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        pre_model.graph.node.append(helper.make_node(
            "Unsqueeze", [name, "tok_unsqueeze_axes"], [name + "_batched"],
            name=f"unsqueeze_{name}"))
    del pre_model.graph.output[:]
    pre_model.graph.output.extend([
        helper.make_tensor_value_info(f"{name}_batched", TensorProto.INT64, [1, None])
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ])
    return pre_model


def _merge_tokenizer_and_classifier(tokenizer_model, classifier_model):
    """Splice the tokenizer branch's outputs into the SetFit classifier's
    exported ONNX graph. Reads the classifier's actual input names rather
    than assuming a fixed layout, since export_onnx's exact naming can
    depend on the installed setfit/skl2onnx version. Requires every
    classifier input to be mapped (not just >=2) -- confirmed by a real
    smoke test that the differentiable-head export takes 3 inputs
    (input_ids, attention_mask, token_type_ids), so silently accepting a
    partial match would leave an unfed graph input."""
    clf_inputs = [i.name for i in classifier_model.graph.input]
    io_map = []
    for clf_name in clf_inputs:
        lower = clf_name.lower()
        if "input_ids" in lower:
            io_map.append(("input_ids_batched", clf_name))
        elif "token_type" in lower:
            io_map.append(("token_type_ids_batched", clf_name))
        elif "attention_mask" in lower or "mask" in lower:
            io_map.append(("attention_mask_batched", clf_name))
    if len(io_map) != len(clf_inputs):
        raise RuntimeError(
            f"couldn't map every classifier input {clf_inputs} to a tokenizer output "
            f"(mapped {[m[1] for m in io_map]}) -- inspect the exported graph and adjust io_map above")
    return onnx.compose.merge_models(
        tokenizer_model, classifier_model, io_map=io_map, prefix1="tok_", prefix2="clf_")


def _pad_to_canonical(model, source_output_name, source_names, order, model_prefix):
    """Take a [1, len(source_names)] output and rebuild it as a full
    [1, len(order)] vector: real value at columns matching source_names
    (by name), zero everywhere else -- mirrors build_regex_graph's own
    zero-fill approach, just scattering the opposite subset of columns."""
    model = onnx.compose.add_prefix(model, prefix=model_prefix)
    source_output_name = model_prefix + source_output_name
    nodes, initializers = list(model.graph.node), list(model.graph.initializer)

    zero_shape_init = helper.make_tensor(f"{model_prefix}one_shape", TensorProto.INT64, [2], [1, 1])
    initializers.append(zero_shape_init)
    col_tensors = []
    for i, name in enumerate(order):
        if name in source_names:
            col_idx = source_names.index(name)
            start = helper.make_tensor(f"{model_prefix}slice_start_{i}", TensorProto.INT64, [1], [col_idx])
            end = helper.make_tensor(f"{model_prefix}slice_end_{i}", TensorProto.INT64, [1], [col_idx + 1])
            axis = helper.make_tensor(f"{model_prefix}slice_axis_{i}", TensorProto.INT64, [1], [1])
            initializers.extend([start, end, axis])
            out_name = f"{model_prefix}col_{i}"
            nodes.append(helper.make_node(
                "Slice", [source_output_name, f"{model_prefix}slice_start_{i}",
                          f"{model_prefix}slice_end_{i}", f"{model_prefix}slice_axis_{i}"],
                [out_name], name=f"{model_prefix}slice_node_{i}"))
        else:
            out_name = f"{model_prefix}zero_{i}"
            nodes.append(helper.make_node(
                "ConstantOfShape", [f"{model_prefix}one_shape"], [out_name],
                value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0]),
                name=f"{model_prefix}zero_const_{i}"))
        col_tensors.append(out_name)

    padded_name = f"{model_prefix}padded_scores"
    nodes.append(helper.make_node("Concat", col_tensors, [padded_name], axis=1,
                                   name=f"{model_prefix}concat_padded"))

    graph = helper.make_graph(
        nodes, f"{model_prefix}padded", model.graph.input,
        [helper.make_tensor_value_info(padded_name, TensorProto.FLOAT, [None, len(order)])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=list(model.opset_import)), padded_name


def _combine_regex_and_semantic(regex_model, semantic_model_padded, semantic_output_name, order):
    """Both graphs take the same `text` input and each already outputs a
    full [1, 33] vector, zero in the columns they don't own -- Add
    reassembles the real vector without any dynamic gather/scatter."""
    nodes = list(regex_model.graph.node) + list(semantic_model_padded.graph.node)
    initializers = list(regex_model.graph.initializer) + list(semantic_model_padded.graph.initializer)
    nodes.append(helper.make_node("Add", ["trope_scores", semantic_output_name], ["final_trope_scores"],
                                   name="combine_regex_and_semantic"))

    opsets = {op.domain: op.version for op in list(regex_model.opset_import) + list(semantic_model_padded.opset_import)}
    graph = helper.make_graph(
        nodes, "merged_tropes",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
        [helper.make_tensor_value_info("final_trope_scores", TensorProto.FLOAT, [None, len(order)])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
    onnx.checker.check_model(model)
    return model


def export_classifiers():
    from runtime.regex_onnx import MECHANICAL_TROPE_NAMES, build_regex_graph

    order = _trope_order()
    regex_model = build_regex_graph(order)
    print(f"regex branch: {len(MECHANICAL_TROPE_NAMES)} mechanical tropes, "
          "verified 0 mismatches vs Python re.search (see runtime/regex_onnx.py)", flush=True)

    if not os.path.isdir(CLASSIFIER_MODEL_DIR):
        print(f"no trained classifier found under {CLASSIFIER_MODEL_DIR}/; "
              "merged model will only cover the regex-detectable tropes", flush=True)
        os.makedirs(ONNX_TROPES_DIR, exist_ok=True)
        # regex_model's output is already named "trope_scores" and already
        # zero-padded for the semantic slots -- usable standalone as-is.
        onnx.save(regex_model, MERGED_MODEL_PATH)
        print(f"saved -> {MERGED_MODEL_PATH} (regex-only)", flush=True)
        return

    from setfit import SetFitModel
    from setfit.exporters.onnx import export_onnx as setfit_export_onnx

    with open(os.path.join(CLASSIFIER_MODEL_DIR, "trope_order.txt"), encoding="utf-8") as fh:
        semantic_names = [line.strip() for line in fh if line.strip()]

    model = SetFitModel.from_pretrained(CLASSIFIER_MODEL_DIR)
    clf_onnx_path = os.path.join(ONNX_TROPES_DIR, "setfit_classifier_raw.onnx")
    os.makedirs(ONNX_TROPES_DIR, exist_ok=True)
    setfit_export_onnx(model.model_body, model.model_head, opset=14, output_path=clf_onnx_path)

    torch_probs = np.array(model.predict_proba([
        "This is an ordinary sentence.",
        "This will change everything for humanity forever.",
    ]))
    print(f"semantic branch: exported -> {clf_onnx_path}  torch_probs sample={torch_probs[:, :2]}", flush=True)

    tokenizer_branch = _build_tokenizer_branch()
    classifier_graph = onnx.load(clf_onnx_path)
    merged_semantic = _merge_tokenizer_and_classifier(tokenizer_branch, classifier_graph)

    # The differentiable head's export produces both raw logits and a
    # precomputed probability tensor under an opaque auto-generated name
    # (confirmed by a real smoke test: outputs were "logits" and "994").
    # Rather than guess which auto-generated name is "the probabilities"
    # across setfit/torch versions, always take the raw logits and apply
    # our own Sigmoid -- one-vs-rest multi-label means each trope's score
    # is independently sigmoided (not a softmax across classes), matching
    # how it was trained (BCEWithLogitsLoss).
    logits_name = next((o.name for o in merged_semantic.graph.output if o.name == "logits"),
                        merged_semantic.graph.output[0].name)
    merged_semantic.graph.node.append(helper.make_node(
        "Sigmoid", [logits_name], ["semantic_probs"], name="semantic_sigmoid"))
    del merged_semantic.graph.output[:]
    merged_semantic.graph.output.append(
        helper.make_tensor_value_info("semantic_probs", TensorProto.FLOAT, [None, len(semantic_names)]))

    padded_semantic, padded_name = _pad_to_canonical(
        merged_semantic, "semantic_probs", semantic_names, order, model_prefix="sm_")
    final_model = _combine_regex_and_semantic(regex_model, padded_semantic, padded_name, order)

    onnx.save(final_model, MERGED_MODEL_PATH)
    print(f"saved -> {MERGED_MODEL_PATH} (regex + semantic merged)", flush=True)


def export_rewriter():
    if not os.path.isdir("models/rewriter"):
        print("no trained rewriter found under models/rewriter/, skipping", flush=True)
        return

    from optimum.exporters.onnx import main_export

    os.makedirs(ONNX_REWRITER_DIR, exist_ok=True)
    main_export(
        model_name_or_path="models/rewriter",
        output=ONNX_REWRITER_DIR,
        task="text2text-generation-with-past",
        opset=17,
    )

    import torch
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("models/rewriter")
    torch_model = AutoModelForSeq2SeqLM.from_pretrained("models/rewriter").eval()
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(ONNX_REWRITER_DIR, provider="CPUExecutionProvider")

    fixture = "remove em-dash addiction: The problem -- and this is the part nobody talks about -- is scale."
    enc = tok(fixture, return_tensors="pt")

    with torch.no_grad():
        torch_ids = torch_model.generate(**enc, max_new_tokens=64, num_beams=1)
    ort_ids = ort_model.generate(**enc, max_new_tokens=64, num_beams=1)

    match = torch.equal(torch_ids, torch.as_tensor(ort_ids))
    print(f"rewriter torch: {tok.decode(torch_ids[0], skip_special_tokens=True)!r}", flush=True)
    print(f"rewriter onnx : {tok.decode(ort_ids[0], skip_special_tokens=True)!r}", flush=True)
    assert match, "ONNX rewriter greedy decode diverges from torch generate()"
    print("REWRITER VALIDATION OK", flush=True)
    tok.save_pretrained(ONNX_REWRITER_DIR)


def main():
    export_classifiers()
    export_rewriter()


if __name__ == "__main__":
    main()
