#!/usr/bin/env python3
"""Safely bypass only the six final output Q/DQ pairs in a split inference graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from stage65b_r2_common import sha256, write_tsv


def shape_text(value: onnx.ValueInfoProto) -> str:
    tensor = value.type.tensor_type
    result: list[str] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            result.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("?")
    return "x".join(result)


def initializer_summary(initializer: onnx.TensorProto) -> tuple[str, str, str]:
    value = numpy_helper.to_array(initializer)
    return (
        TensorProto.DataType.Name(initializer.data_type),
        "x".join(map(str, value.shape)) if value.shape else "scalar",
        f"{float(value.min())}:{float(value.max())}",
    )


def initializer_payload_identity(initializer: onnx.TensorProto) -> tuple[str, str, str]:
    value = numpy_helper.to_array(initializer)
    return (
        str(value.dtype),
        "x".join(map(str, value.shape)) if value.shape else "scalar",
        hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tail", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    options = parser.parse_args()
    if options.output.exists() or options.evidence_dir.exists():
        raise RuntimeError("refusing to overwrite D8 output or evidence")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.evidence_dir.mkdir(parents=True)

    model = onnx.load(options.input, load_external_data=False)
    tail = onnx.load(options.tail, load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node)
    initializers = {item.name: item for item in model.graph.initializer}
    output_names = [item.name for item in model.graph.output]
    tail_inputs = [item.name for item in tail.graph.input]
    if len(output_names) != 6 or output_names != tail_inputs:
        raise ValueError("D8 requires six outputs exactly matching tail inputs")

    topology_rows: list[dict[str, Any]] = []
    removed: set[int] = set()
    additions: list[onnx.NodeProto] = []
    diff_rows: list[dict[str, Any]] = []
    for index, output_info in enumerate(model.graph.output):
        boundary = output_info.name
        dq = producers.get(boundary)
        if dq is None or dq.op_type != "DequantizeLinear":
            raise ValueError(f"boundary lacks direct DequantizeLinear producer: {boundary}")
        if len(dq.input) < 3:
            raise ValueError(f"DequantizeLinear lacks explicit scale/ZP: {boundary}")
        q = producers.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear" or len(q.input) < 3:
            raise ValueError(f"boundary lacks exact Q->DQ topology: {boundary}")
        if len(consumers[q.output[0]]) != 1 or consumers[q.output[0]][0] is not dq:
            raise ValueError(f"shared QuantizeLinear output at boundary: {boundary}")
        if consumers.get(boundary):
            raise ValueError(f"graph output also has internal consumers: {boundary}")
        if q.input[1] not in initializers or q.input[2] not in initializers:
            raise ValueError(f"Q scale/ZP are not static initializers: {boundary}")
        if dq.input[1] not in initializers or dq.input[2] not in initializers:
            raise ValueError(f"DQ scale/ZP are not static initializers: {boundary}")
        q_scale_dtype, q_scale_shape, q_scale_range = initializer_summary(
            initializers[q.input[1]]
        )
        q_zp_dtype, q_zp_shape, q_zp_range = initializer_summary(
            initializers[q.input[2]]
        )
        q_scale_identity = initializer_payload_identity(initializers[q.input[1]])
        dq_scale_identity = initializer_payload_identity(initializers[dq.input[1]])
        q_zp_identity = initializer_payload_identity(initializers[q.input[2]])
        dq_zp_identity = initializer_payload_identity(initializers[dq.input[2]])
        if q_scale_identity != dq_scale_identity or q_zp_identity != dq_zp_identity:
            raise ValueError(f"Q/DQ qparam payload mismatch: {boundary}")
        float_source = q.input[0]
        identity = helper.make_node(
            "Identity",
            [float_source],
            [boundary],
            name=f"Stage65B_R2_D8_OutputIdentity_{index}",
        )
        additions.append(identity)
        removed.update((id(q), id(dq)))
        topology_rows.append(
            {
                "boundary_index": index,
                "boundary_name": boundary,
                "shape": shape_text(output_info),
                "dtype": TensorProto.DataType.Name(output_info.type.tensor_type.elem_type),
                "dq_node": dq.name,
                "q_node": q.name,
                "float_source": float_source,
                "float_source_producer": producers[float_source].name
                if float_source in producers
                else "graph-input-or-initializer",
                "float_source_other_consumers": len(consumers[float_source]) - 1,
                "q_output_consumers": len(consumers[q.output[0]]),
                "dq_output_consumers": len(consumers.get(boundary, [])),
                "q_scale_dtype": q_scale_dtype,
                "q_scale_shape": q_scale_shape,
                "q_scale_min_max": q_scale_range,
                "q_scale_initializer": q.input[1],
                "dq_scale_initializer": dq.input[1],
                "scale_payload_sha256": q_scale_identity[2],
                "q_zero_point_dtype": q_zp_dtype,
                "q_zero_point_shape": q_zp_shape,
                "q_zero_point_min_max": q_zp_range,
                "q_zero_point_initializer": q.input[2],
                "dq_zero_point_initializer": dq.input[2],
                "zero_point_payload_sha256": q_zp_identity[2],
                "q_dq_qparam_payload_identity": "pass",
                "shared_node_status": "unshared",
                "topology_status": "pass",
            }
        )
        diff_rows.append(
            {
                "boundary": boundary,
                "removed_quantize": q.name,
                "removed_dequantize": dq.name,
                "added_identity": identity.name,
                "identity_input": float_source,
                "identity_output": boundary,
            }
        )

    before_nodes = len(model.graph.node)
    before_qdq = sum(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in model.graph.node
    )
    retained = [node for node in model.graph.node if id(node) not in removed]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    model.graph.node.extend(additions)
    model.doc_string = (
        model.doc_string
        + "\nStage65B-R2 D8: host-only diagnostic bypass of six final output Q/DQ pairs."
    ).strip()
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, options.output)
    reloaded = onnx.load(options.output, load_external_data=False)
    onnx.checker.check_model(reloaded)

    after_outputs = [item.name for item in reloaded.graph.output]
    after_shapes = [shape_text(item) for item in reloaded.graph.output]
    after_types = [item.type.tensor_type.elem_type for item in reloaded.graph.output]
    qlinear_count = sum(
        node.op_type.startswith("QLinear") for node in reloaded.graph.node
    )
    qdq_count = sum(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in reloaded.graph.node
    )
    conv_failures = 0
    initializer_shapes = {
        item.name: tuple(item.dims) for item in reloaded.graph.initializer
    }
    reloaded_producers = {
        output: node for node in reloaded.graph.node for output in node.output
    }

    def static_weight_shape(value_name: str) -> tuple[int, ...] | None:
        visited: set[str] = set()
        while value_name not in visited:
            visited.add(value_name)
            if value_name in initializer_shapes:
                return initializer_shapes[value_name]
            producer = reloaded_producers.get(value_name)
            if producer is None or producer.op_type not in {
                "DequantizeLinear",
                "QuantizeLinear",
                "Identity",
            }:
                return None
            value_name = producer.input[0]
        return None

    for node in reloaded.graph.node:
        if node.op_type != "Conv":
            continue
        attributes = {item.name: item for item in node.attribute}
        weight_shape = static_weight_shape(node.input[1])
        expected = list(weight_shape[-2:]) if weight_shape and len(weight_shape) >= 2 else None
        observed = list(attributes["kernel_shape"].ints) if "kernel_shape" in attributes else None
        conv_failures += int(expected is None or observed != expected)

    uint8_zero_points = 0
    for node in reloaded.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
            continue
        initializer = initializers.get(node.input[2])
        if initializer is not None and initializer.data_type == TensorProto.UINT8:
            uint8_zero_points += 1

    status = all(
        (
            after_outputs == output_names,
            after_shapes == [shape_text(item) for item in model.graph.output],
            after_types == [item.type.tensor_type.elem_type for item in model.graph.output],
            qlinear_count == 0,
            uint8_zero_points == 0,
            conv_failures == 0,
            qdq_count == before_qdq - 12,
        )
    )
    write_tsv(options.evidence_dir / "d8_topology.tsv", topology_rows)
    (options.evidence_dir / "d8_graph_diff.json").write_text(
        json.dumps(
            {
                "classification": "surgically-modified-host-diagnostic-only",
                "deployable_or_provider_evidence": False,
                "input_model": str(options.input),
                "input_sha256": sha256(options.input),
                "output_model": str(options.output),
                "removed": diff_rows,
                "unchanged_upstream_qdq_count": qdq_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    identity = {
        "classification": "surgically-modified-host-diagnostic-only",
        "input_sha256": sha256(options.input),
        "output_sha256": sha256(options.output),
        "tail_sha256": sha256(options.tail),
        "nodes_before": before_nodes,
        "nodes_after": len(reloaded.graph.node),
        "qdq_before": before_qdq,
        "qdq_after": qdq_count,
        "removed_final_qdq_nodes": 12,
        "added_identity_nodes": 6,
    }
    write_tsv(options.evidence_dir / "d8_model_identity.tsv", [identity])
    conformance = {
        "checker": "pass",
        "shape_inference": "pass",
        "six_outputs": int(len(after_outputs) == 6),
        "tail_input_identity": int(after_outputs == tail_inputs),
        "output_name_identity": int(after_outputs == output_names),
        "output_shape_identity": int(
            after_shapes == [shape_text(item) for item in model.graph.output]
        ),
        "output_dtype_float": int(
            all(item == TensorProto.FLOAT for item in after_types)
        ),
        "qlinear_count": qlinear_count,
        "uint8_zero_point_count": uint8_zero_points,
        "conv_kernel_shape_failures": conv_failures,
        "status": "pass" if status else "fail",
    }
    write_tsv(options.evidence_dir / "d8_conformance.tsv", [conformance])
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
