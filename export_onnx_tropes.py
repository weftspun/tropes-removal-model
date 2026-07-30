# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""
Build the ONE merged ONNX model gate.py loads: single input `text`
(string[1] -- one sentence per call), single output `trope_scores`
(float32[1, N], canonical STE violation order from seeds/trope.parquet).

Two branches, assembled into one graph:
  1. The regex branch (runtime/regex_onnx.py) -- ~18 mechanical STE
     violations, deterministic RegexFullMatch nodes, zero training data.
     VERIFIED: 0 mismatches vs Python's re.search across 30,000 real
     sentences. Its output is already a full N-column vector with zeros
     in the semantic slots.
  2. The semantic branch -- a tokenizer (onnxruntime_extensions,
     auto-generated from the SetFit body's own tokenizer) spliced onto the
     SetFit classifier exported with a differentiable head.
     Its output is padded into the same N-column shape, zeros in the
     mechanical slots.

Both branches consume the same `text` input and are additively combined
(`Add`): since each column is zero in exactly one branch, elementwise sum
reassembles the full vector without any dynamic gather/scatter.
`onnx.compose.add_prefix` avoids node/tensor name collisions between the
two independently-built subgraphs before they're spliced into one GraphProto.

Also exports the flan-t5-small rewriter separately (onnx_rewriter/) --
that one stays a standalone seq2seq model; a generation model doesn't fold
into the same "one classification graph" simplification as the two branches
above, and optimum's seq2seq export already produces its own encoder/decoder
pair.
"""
import os
import warnings

import numpy as np
import onnx
import onnx.version_converter
from onnx import TensorProto, helper

warnings.filterwarnings("ignore")

ONNX_VIOLATIONS_DIR = "onnx_violations"
ONNX_REWRITER_DIR = "onnx_rewriter"
MERGED_MODEL_PATH = os.path.join(ONNX_VIOLATIONS_DIR, "merged_model.onnx")
CLASSIFIER_MODEL_DIR = "models/setfit_classifier"
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # must match train_tropes.py's SetFit body


def _trope_order():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from runtime.datalake import read_tropes
    return read_tropes().to_pandas()["name"].tolist()


def _build_tokenizer_branch():
    """text (string[1]) -> input_ids, attention_mask, token_type_ids (int64[1, seq_len])."""
    import onnxruntime_extensions as ortx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    pre_model, _ = ortx.gen_processing_models(
        tok, pre_kwargs={"WITH_DEFAULT_INPUTS": True, "CAST_TOKEN_ID": True})

    onnx_domain_version = next(
        (op.version for op in pre_model.opset_import if op.domain in ("", "ai.onnx")), 18)
    pre_model.opset_import.append(helper.make_opsetid("", onnx_domain_version))

    axes_init = helper.make_tensor("tok_unsqueeze_axes", TensorProto.INT64, [1], [0])
    pre_model.graph.initializer.append(axes_init)
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
    """Splice tokenizer outputs into the SetFit classifier's ONNX graph."""
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
    target_opset = next(op.version for op in tokenizer_model.opset_import if op.domain == "")
    classifier_model = onnx.version_converter.convert_version(classifier_model, target_opset)
    ir_version = max(tokenizer_model.ir_version, classifier_model.ir_version)
    tokenizer_model.ir_version = ir_version
    classifier_model.ir_version = ir_version
    return onnx.compose.merge_models(
        tokenizer_model, classifier_model, io_map=io_map, prefix1="tok_", prefix2="clf_")


def _pad_to_canonical(model, source_output_name, source_names, order, model_prefix):
    """Pad a [1, len(source_names)] output to full [1, len(order)] vector,
    zeros for names not in source_names."""
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


def _rename_tensor(graph, old_name, new_name):
    """Rename a tensor throughout a graph in place."""
    for collection in (graph.input, graph.output):
        for value_info in collection:
            if value_info.name == old_name:
                value_info.name = new_name
    for initializer in graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name
    for node in graph.node:
        node.input[:] = [new_name if n == old_name else n for n in node.input]
        node.output[:] = [new_name if n == old_name else n for n in node.output]


def _combine_regex_and_semantic(regex_model, semantic_model_padded, semantic_output_name, order):
    """Both graphs output a full [1, N] vector, zero in opposite columns --
    Add reassembles them."""
    nodes = list(regex_model.graph.node) + list(semantic_model_padded.graph.node)
    initializers = list(regex_model.graph.initializer) + list(semantic_model_padded.graph.initializer)
    nodes.append(helper.make_node("Add", ["trope_scores", semantic_output_name], ["final_trope_scores"],
                                   name="combine_regex_and_semantic"))

    opsets = {}
    for op in list(regex_model.opset_import) + list(semantic_model_padded.opset_import):
        domain = "" if op.domain == "ai.onnx" else op.domain
        opsets[domain] = max(op.version, opsets.get(domain, 0))
    graph = helper.make_graph(
        nodes, "merged_ste_violations",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [None])],
        [helper.make_tensor_value_info("final_trope_scores", TensorProto.FLOAT, [None, len(order)])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
    onnx.checker.check_model(model)
    return model


def _export_setfit_classifier_to_onnx(model, output_path, opset):
    """Export the SetFit differentiable-head classifier ourselves instead of
    calling setfit.exporters.onnx.export_onnx(): that function's
    OnnxSetFitModel wrapper only chains model_body._modules[\"0\"] (transformer)
    and [\"1\"] (pooling) before the head, silently dropping module \"2\"
    (Normalize -- L2-normalizes the pooled embedding to unit length)."""
    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, body, head):
            super().__init__()
            self.body = body
            self.head = head

        def forward(self, input_ids, attention_mask, token_type_ids):
            features = {"input_ids": input_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids}
            for module in self.body._modules.values():
                features = module(features)
            return self.head(features["sentence_embedding"])

    body_transformer = model.model_body._modules["0"]
    tokenizer = body_transformer.tokenizer
    max_length = body_transformer.max_seq_length
    dummy = tokenizer("It's a test.", max_length=max_length, padding="max_length",
                       return_attention_mask=True, return_token_type_ids=True, return_tensors="pt")

    wrapper = _Wrapper(model.model_body, model.model_head).eval().cpu()
    dummy = {k: v.cpu() for k, v in dummy.items()}
    with torch.no_grad():
        torch.onnx.export(
            wrapper, args=tuple(dummy.values()), f=output_path, opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "token_type_ids": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
        )


def export_classifiers():
    from runtime.regex_onnx import MECHANICAL_VIOLATION_NAMES, build_regex_graph

    order = _trope_order()
    regex_model = build_regex_graph(order)
    print(f"regex branch: {len(MECHANICAL_VIOLATION_NAMES)} mechanical STE violations, "
          "verified 0 mismatches vs Python re.search (see runtime/regex_onnx.py)", flush=True)

    if not os.path.isdir(CLASSIFIER_MODEL_DIR):
        print(f"no trained classifier found under {CLASSIFIER_MODEL_DIR}/; "
              "merged model will only cover the regex-detectable violations", flush=True)
        os.makedirs(ONNX_VIOLATIONS_DIR, exist_ok=True)
        onnx.save(regex_model, MERGED_MODEL_PATH)
        print(f"saved -> {MERGED_MODEL_PATH} (regex-only)", flush=True)
        return

    from setfit import SetFitModel

    with open(os.path.join(CLASSIFIER_MODEL_DIR, "trope_order.txt"), encoding="utf-8") as fh:
        semantic_names = [line.strip() for line in fh if line.strip()]

    model = SetFitModel.from_pretrained(CLASSIFIER_MODEL_DIR)
    clf_onnx_path = os.path.join(ONNX_VIOLATIONS_DIR, "setfit_classifier_raw.onnx")
    os.makedirs(ONNX_VIOLATIONS_DIR, exist_ok=True)
    _export_setfit_classifier_to_onnx(model, clf_onnx_path, opset=14)

    tokenizer_branch = _build_tokenizer_branch()
    classifier_graph = onnx.load(clf_onnx_path)
    merged_semantic = _merge_tokenizer_and_classifier(tokenizer_branch, classifier_graph)

    logits_name = next((o.name for o in merged_semantic.graph.output if o.name == "logits"),
                        merged_semantic.graph.output[0].name)
    merged_semantic.graph.node.append(helper.make_node(
        "Sigmoid", [logits_name], ["semantic_probs"], name="semantic_sigmoid"))
    del merged_semantic.graph.output[:]
    merged_semantic.graph.output.append(
        helper.make_tensor_value_info("semantic_probs", TensorProto.FLOAT, [None, len(semantic_names)]))

    padded_semantic, padded_name = _pad_to_canonical(
        merged_semantic, "semantic_probs", semantic_names, order, model_prefix="sm_")
    [text_input] = padded_semantic.graph.input
    _rename_tensor(padded_semantic.graph, text_input.name, "text")
    final_model = _combine_regex_and_semantic(regex_model, padded_semantic, padded_name, order)

    onnx.save(final_model, MERGED_MODEL_PATH)
    print(f"saved -> {MERGED_MODEL_PATH} (regex + semantic merged)", flush=True)

    thresholds_src = os.path.join(CLASSIFIER_MODEL_DIR, "thresholds.json")
    if os.path.isfile(thresholds_src):
        import shutil
        shutil.copy(thresholds_src, os.path.join(ONNX_VIOLATIONS_DIR, "thresholds.json"))
        print(f"copied -> {ONNX_VIOLATIONS_DIR}/thresholds.json", flush=True)


def export_rewriter():
    if not os.path.isdir("models/rewriter"):
        print("no trained rewriter found under models/rewriter/, skipping", flush=True)
        return

    import shutil
    from optimum.exporters.onnx import main_export

    shutil.rmtree(ONNX_REWRITER_DIR, ignore_errors=True)
    os.makedirs(ONNX_REWRITER_DIR, exist_ok=True)
    main_export(
        model_name_or_path="models/rewriter",
        output=ONNX_REWRITER_DIR,
        task="text2text-generation",
        opset=17,
    )

    import torch
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("models/rewriter")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        ONNX_REWRITER_DIR, provider="CPUExecutionProvider", use_cache=False)

    fixture = "remove passive voice: The file is read by the parser."
    enc = tok(fixture, return_tensors="pt")

    decoder_start_id = ort_model.config.decoder_start_token_id
    eos_id = ort_model.config.eos_token_id
    decoder_input_ids = torch.tensor([[decoder_start_id]])
    ort_generated = []
    with torch.no_grad():
        for _ in range(64):
            out = ort_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids, use_cache=False)
            next_id = out.logits[:, -1].argmax(dim=-1)
            if next_id.item() == eos_id:
                break
            ort_generated.append(next_id.item())
            decoder_input_ids = torch.cat([decoder_input_ids, next_id[:, None]], dim=-1)

    print(f"rewriter onnx : {tok.decode(ort_generated, skip_special_tokens=True)!r}", flush=True)
    tok.save_pretrained(ONNX_REWRITER_DIR)


def main():
    export_classifiers()
    export_rewriter()


if __name__ == "__main__":
    main()