#!/usr/bin/env python3
"""Focused positive and fail-closed tests for Stage65B-R3 tooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from stage65b_r3_common import extract_model, validate_extraction_contract
from stage65b_r3_evaluate import compare
from stage65b_r3_localizer import (
    FrontierSpec,
    complete_cut,
    partition_for_spec,
    semantic_mapping,
    validate_complete_cut,
)


def graph_model(nodes, inputs, outputs, initializers=(), value_info=(), opset=18):
    graph = helper.make_graph(
        nodes,
        "test",
        inputs,
        outputs,
        initializer=list(initializers),
        value_info=list(value_info),
    )
    return helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", opset)],
    )


def branched_graph(op="Add"):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])
        for name in ("a", "b", "c")
    ]
    nodes = [
        helper.make_node("Identity", ["x"], ["a"], name="n0"),
        helper.make_node("Identity", ["a"], ["b"], name="n1"),
        helper.make_node("Identity", ["a"], ["c"], name="n2"),
        helper.make_node(op, ["b", "c"], ["y"], name="merge"),
    ]
    return graph_model(nodes, [x], [y], value_info=infos)


def source_identity(shape=(1, 2), dtype=TensorProto.FLOAT):
    x = helper.make_tensor_value_info("x", dtype, list(shape))
    y = helper.make_tensor_value_info("semantic", dtype, list(shape))
    node = helper.make_node("Identity", ["x"], ["semantic"], name="source")
    return graph_model([node], [x], [y])


def qdq_candidate(
    shape=(1, 2),
    dtype=TensorProto.FLOAT,
    duplicate_source=False,
    shared_q=False,
):
    x = helper.make_tensor_value_info("x", dtype, list(shape))
    y = helper.make_tensor_value_info("semantic", dtype, list(shape))
    scale = numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "scale")
    zp = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zp")
    nodes = [
        helper.make_node("Identity", ["x"], ["raw"], name="source"),
        helper.make_node("QuantizeLinear", ["raw", "scale", "zp"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["semantic"], name="dq"),
    ]
    outputs = [y]
    if duplicate_source:
        nodes.insert(1, helper.make_node("Identity", ["x"], ["raw2"], name="source"))
    if shared_q:
        other = helper.make_tensor_value_info("other", TensorProto.FLOAT, list(shape))
        nodes.append(
            helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["other"], name="dq2")
        )
        outputs.append(other)
    return graph_model(nodes, [x], outputs, [scale, zp])


def test_incomplete_multi_tensor_cut_rejected():
    model = branched_graph("Add")
    upstream, downstream = {0, 1}, {2, 3}
    assert complete_cut(model, upstream, downstream) == ["a", "b"]
    with pytest.raises(ValueError, match="incomplete cut"):
        validate_complete_cut(model, upstream, downstream, ["b"])


def test_hidden_concat_edge_rejected():
    model = branched_graph("Concat")
    upstream, downstream = {0, 1}, {2, 3}
    with pytest.raises(ValueError, match="incomplete cut"):
        validate_complete_cut(model, upstream, downstream, ["b"])


def test_ambiguous_source_qdq_mapping_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        semantic_mapping(source_identity(), qdq_candidate(duplicate_source=True), "semantic")


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        semantic_mapping(source_identity((1, 2)), qdq_candidate((1, 3)), "semantic")


def test_dtype_mismatch_rejected():
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        semantic_mapping(
            source_identity((1, 2), TensorProto.DOUBLE),
            qdq_candidate((1, 2), TensorProto.FLOAT),
            "semantic",
        )


def test_dynamic_cut_shape_rejected():
    dynamic = source_identity((1, "dynamic"))
    with pytest.raises(ValueError, match="dynamic or unresolved"):
        semantic_mapping(dynamic, qdq_candidate((1, 2)), "semantic")


def test_duplicate_tensor_name_rejected():
    model = branched_graph()
    with pytest.raises(ValueError, match="duplicate extraction input"):
        extract_model(model, ["a", "a"], ["y"])


def test_missing_initializer_or_value_rejected():
    model = branched_graph()
    model.graph.node[3].input.append("not_present")
    with pytest.raises(ValueError, match="missing initializer or graph value"):
        validate_complete_cut(model, {0, 1, 2}, {3}, ["b", "c"])


def test_functionproto_and_opset_preserved():
    function = helper.make_function(
        "local",
        "Twice",
        ["X"],
        ["Y"],
        [helper.make_node("Add", ["X", "X"], ["Y"])],
        [helper.make_opsetid("", 18)],
    )
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    model = graph_model(
        [helper.make_node("Twice", ["x"], ["y"], domain="local", name="twice")],
        [x],
        [y],
    )
    model.opset_import.extend([helper.make_opsetid("local", 1)])
    model.functions.extend([function])
    extracted = extract_model(model, ["x"], ["y"])
    assert [(item.domain, item.version) for item in extracted.opset_import] == [
        ("", 18),
        ("local", 1),
    ]
    assert [(item.domain, item.name) for item in extracted.functions] == [("local", "Twice")]
    broken = onnx.ModelProto()
    broken.CopyFrom(extracted)
    del broken.functions[:]
    with pytest.raises(ValueError, match="lost required FunctionProto"):
        validate_extraction_contract(model, broken)
    broken_opset = onnx.ModelProto()
    broken_opset.CopyFrom(extracted)
    broken_opset.opset_import[0].version = 17
    with pytest.raises(ValueError, match="opset imports changed"):
        validate_extraction_contract(model, broken_opset)


def test_control_reconstruction_mismatch_detected():
    left = np.asarray([[1.0, 2.0]], dtype=np.float32)
    right = np.asarray([[1.0, 2.5]], dtype=np.float32)
    result = compare(left, right)
    assert result["byte_identical"] == 0
    assert result["max_abs"] == pytest.approx(0.5)


def test_unsafe_shared_qdq_node_rejected():
    with pytest.raises(ValueError, match="unsafe shared QuantizeLinear"):
        semantic_mapping(source_identity(), qdq_candidate(shared_q=True), "semantic")


def test_multi_anchor_partition_uses_complete_ancestor_closure():
    model = graph_model(
        [
            helper.make_node("Identity", ["x"], ["stem"], name="stem"),
            helper.make_node("Identity", ["stem"], ["left"], name="left"),
            helper.make_node("Identity", ["stem"], ["right"], name="right"),
            helper.make_node("Add", ["left", "right"], ["y"], name="join"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    upstream, downstream = partition_for_spec(
        model, FrontierSpec("T", "test", "left;right", "multi-anchor")
    )
    cut = complete_cut(model, upstream, downstream)
    assert set(cut) == {"left", "right"}
    validate_complete_cut(model, upstream, downstream, cut)
