#!/usr/bin/env python3
"""Plan and build safe Stage65B-R3 source/QDQ cut frontiers."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import ModelProto, TensorProto, numpy_helper

from stage65b_r3_common import (
    consumers,
    dtype_text,
    extract_model,
    inferred_model,
    model_manifest,
    node_name_map,
    producers,
    sha256,
    shape_text,
    static_tensor,
    value_info_map,
    write_tsv,
)


@dataclass(frozen=True)
class FrontierSpec:
    frontier: str
    region: str
    anchor: str
    mode: str = "cutoff"


COARSE_SPECS = (
    FrontierSpec("C0", "R0-stem-model0-model2", "/model.2/cv2/act/Mul"),
    FrontierSpec("C1", "R1-early-mid-backbone-model3-model6", "/model.6/cv2/act/Mul"),
    FrontierSpec("C2", "R2-later-backbone-model7-model8", "/model.8/cv2/act/Mul"),
    FrontierSpec("C3", "R3-model9-model10-attention-ffn", "/model.10/cv2/act/Mul"),
    FrontierSpec("C4", "R4-neck-entry-model11-model13", "/model.13/cv2/act/Mul"),
    FrontierSpec("C5", "R5-neck-fusion-model14-model20", "/model.20/act/Mul"),
    FrontierSpec("C6", "R6-pre-head-model21-model22", "/model.22/cv2/act/Mul"),
    FrontierSpec("C7", "R7-per-scale-head-prefixes", "six-final-head-convs", "head"),
)


REGIONS = (
    ("R0", "input/stem and early model.0-model.2", 0, 2),
    ("R1", "early/mid backbone model.3-model.6", 3, 6),
    ("R2", "later backbone model.7-model.8", 7, 8),
    ("R3", "model.9 and model.10 attention/FFN", 9, 10),
    ("R4", "late backbone / neck entry model.11-model.13", 11, 13),
    ("R5", "neck fusion model.14-model.20", 14, 20),
    ("R6", "model.21/model.22 pre-head, Concat/Split and attention/FFN", 21, 22),
    ("R7", "model.23 per-scale head prefixes and terminal Conv", 23, 23),
)


def module_number(name: str) -> int | None:
    match = re.search(r"/model\.(\d+)(?:/|$)", name)
    return int(match.group(1)) if match else None


def region_for_node(name: str) -> str:
    module = module_number(name)
    if module is None:
        return "non-module-constant-or-helper"
    for region, _, first, last in REGIONS:
        if first <= module <= last:
            return region
    return "unmapped"


def producer_index(model: ModelProto) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            if output in result:
                raise ValueError(f"duplicate tensor producer: {output}")
            result[output] = index
    return result


def consumer_indices(model: ModelProto) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, node in enumerate(model.graph.node):
        for value in node.input:
            if value:
                result.setdefault(value, []).append(index)
    return result


def partition_for_spec(model: ModelProto, spec: FrontierSpec) -> tuple[set[int], set[int]]:
    if spec.mode == "cutoff":
        matches = [index for index, node in enumerate(model.graph.node) if node.name == spec.anchor]
        if len(matches) != 1:
            raise ValueError(
                f"{spec.frontier}: anchor must match exactly one node: {spec.anchor} ({len(matches)})"
            )
        upper = matches[0]
        upstream = set(range(upper + 1))
    elif spec.mode == "head":
        output_names = {item.name for item in model.graph.output}
        terminal = {
            index
            for index, node in enumerate(model.graph.node)
            if any(output in output_names for output in node.output)
        }
        if len(terminal) != 6 or any(
            model.graph.node[index].op_type != "Conv" for index in terminal
        ):
            raise ValueError("head frontier requires exactly six terminal Conv producers")
        upstream = set(range(len(model.graph.node))) - terminal
    elif spec.mode == "multi-anchor":
        anchors = [item for item in spec.anchor.split(";") if item]
        if not anchors or len(anchors) != len(set(anchors)):
            raise ValueError(f"{spec.frontier}: multi-anchor names must be unique")
        by_name = {
            node.name: index for index, node in enumerate(model.graph.node) if node.name
        }
        missing = [name for name in anchors if name not in by_name]
        if missing:
            raise ValueError(f"{spec.frontier}: missing multi-anchor nodes: {missing}")
        produced = producer_index(model)
        pending = [by_name[name] for name in anchors]
        upstream: set[int] = set()
        while pending:
            index = pending.pop()
            if index in upstream:
                continue
            upstream.add(index)
            for value in model.graph.node[index].input:
                if value and value in produced:
                    pending.append(produced[value])
    else:
        raise ValueError(f"unsupported frontier mode: {spec.mode}")
    downstream = set(range(len(model.graph.node))) - upstream
    # ONNX Constant nodes are immutable graph constants, not mutable frontier
    # state. Keep a Constant in the suffix when none of its consumers is in the
    # upstream partition; Extractor will retain it where referenced.
    indexed_consumers = consumer_indices(model)
    move_to_downstream = {
        index
        for index in upstream
        if model.graph.node[index].op_type == "Constant"
        and not any(
            consumer in upstream
            for output in model.graph.node[index].output
            for consumer in indexed_consumers.get(output, [])
        )
    }
    upstream -= move_to_downstream
    downstream |= move_to_downstream
    if not upstream or not downstream:
        raise ValueError(f"{spec.frontier}: empty graph partition")
    return upstream, downstream


def complete_cut(
    model: ModelProto, upstream: set[int], downstream: set[int]
) -> list[str]:
    indexed_consumers = consumer_indices(model)
    cut: list[str] = []
    for index, node in enumerate(model.graph.node):
        if index not in upstream:
            continue
        if node.op_type == "Constant":
            continue
        for output in node.output:
            if any(index in downstream for index in indexed_consumers.get(output, [])):
                cut.append(output)
    if not cut:
        raise ValueError("frontier has no live cross-partition tensors")
    if len(cut) != len(set(cut)):
        raise ValueError("frontier has duplicate tensor names")
    return cut


def validate_complete_cut(
    model: ModelProto,
    upstream: set[int],
    downstream: set[int],
    cut: list[str],
) -> None:
    produced = producer_index(model)
    initializers = {item.name for item in model.graph.initializer}
    graph_inputs = {item.name for item in model.graph.input}
    expected: set[str] = set()
    for index in downstream:
        for value in model.graph.node[index].input:
            if not value:
                continue
            source = produced.get(value)
            if source in upstream and model.graph.node[source].op_type != "Constant":
                expected.add(value)
            elif source is None and value not in initializers and value not in graph_inputs:
                raise ValueError(f"missing initializer or graph value: {value}")
            elif source is None and value in graph_inputs:
                raise ValueError(f"graph input crosses the frontier directly: {value}")
    if expected != set(cut):
        missing = sorted(expected - set(cut))
        extra = sorted(set(cut) - expected)
        raise ValueError(f"incomplete cut set; missing={missing}, extra={extra}")


def qparam_summary(initializer: onnx.TensorProto) -> tuple[str, str, str]:
    values = numpy_helper.to_array(initializer)
    shape = "scalar" if values.ndim == 0 else "x".join(map(str, values.shape))
    minimum = float(values.min()) if values.size else float("nan")
    maximum = float(values.max()) if values.size else float("nan")
    return TensorProto.DataType.Name(initializer.data_type), shape, f"{minimum}:{maximum}"


def semantic_mapping(
    source: ModelProto,
    candidate: ModelProto,
    tensor: str,
) -> dict[str, Any]:
    source_producers = producers(source)
    source_info = value_info_map(source)
    candidate_info = value_info_map(candidate)
    candidate_names = node_name_map(candidate)
    candidate_consumers = consumers(candidate)
    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    source_node = source_producers.get(tensor)
    if source_node is None:
        raise ValueError(f"source tensor has no node producer: {tensor}")
    output_indices = [index for index, output in enumerate(source_node.output) if output == tensor]
    if len(output_indices) != 1:
        raise ValueError(f"ambiguous source output index: {tensor}")
    output_index = output_indices[0]
    matches = [
        node
        for node in candidate_names.get(source_node.name, [])
        if node.op_type == source_node.op_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source/QDQ node mapping is ambiguous: {source_node.name} ({len(matches)})"
        )
    candidate_node = matches[0]
    if output_index >= len(candidate_node.output):
        raise ValueError(f"candidate producer lacks output index {output_index}: {source_node.name}")
    raw = candidate_node.output[output_index]
    q_matches = [item for item in candidate_consumers.get(raw, []) if item.op_type == "QuantizeLinear"]
    if len(q_matches) != 1:
        raise ValueError(f"expected one QuantizeLinear after {source_node.name}: {len(q_matches)}")
    q_node = q_matches[0]
    if len(q_node.input) < 3 or len(q_node.output) != 1:
        raise ValueError(f"QuantizeLinear lacks explicit qparams: {q_node.name}")
    if len(candidate_consumers.get(q_node.output[0], [])) != 1:
        raise ValueError(f"unsafe shared QuantizeLinear output: {q_node.name}")
    dq_node = candidate_consumers[q_node.output[0]][0]
    if dq_node.op_type != "DequantizeLinear" or len(dq_node.output) != 1:
        raise ValueError(f"expected direct DequantizeLinear after {q_node.name}")
    mapped = dq_node.output[0]
    if mapped != tensor:
        raise ValueError(f"semantic tensor name drift: source={tensor}, candidate={mapped}")
    if tensor not in source_info or mapped not in candidate_info:
        raise ValueError(f"missing shape information for mapped tensor: {tensor}")
    left, right = source_info[tensor], candidate_info[mapped]
    if not static_tensor(left) or not static_tensor(right):
        raise ValueError(f"dynamic or unresolved cut shape: {tensor}")
    if shape_text(left) != shape_text(right) or dtype_text(left) != dtype_text(right):
        raise ValueError(
            f"shape/dtype mismatch for {tensor}: "
            f"{shape_text(left)}/{dtype_text(left)} vs {shape_text(right)}/{dtype_text(right)}"
        )
    if q_node.input[1] not in candidate_initializers or q_node.input[2] not in candidate_initializers:
        raise ValueError(f"non-static qparams at frontier: {tensor}")
    scale_dtype, scale_shape, scale_range = qparam_summary(candidate_initializers[q_node.input[1]])
    zp_dtype, zp_shape, zp_range = qparam_summary(candidate_initializers[q_node.input[2]])
    axis = next((item.i for item in q_node.attribute if item.name == "axis"), "")
    return {
        "source_tensor": tensor,
        "source_producer": source_node.name,
        "source_op": source_node.op_type,
        "source_output_index": output_index,
        "shape": shape_text(left),
        "dtype": dtype_text(left),
        "candidate_producer": candidate_node.name,
        "candidate_raw_tensor": raw,
        "quantize_node": q_node.name,
        "dequantize_node": dq_node.name,
        "candidate_float_tensor": mapped,
        "scale_initializer": q_node.input[1],
        "scale_dtype": scale_dtype,
        "scale_shape": scale_shape,
        "scale_min_max": scale_range,
        "zero_point_initializer": q_node.input[2],
        "zero_point_dtype": zp_dtype,
        "zero_point_shape": zp_shape,
        "zero_point_min_max": zp_range,
        "axis": axis,
        "q_output_consumer_count": len(candidate_consumers[q_node.output[0]]),
        "mapping_status": "pass",
    }


def save_model(model: ModelProto, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, path)
    loaded = inferred_model(path)
    return model_manifest(path, loaded)


def build_splits(
    fp32: ModelProto,
    candidate: ModelProto,
    d8: ModelProto,
    spec: FrontierSpec,
    cut: list[str],
    upstream: set[int],
    downstream: set[int],
    model_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_inputs = [item.name for item in fp32.graph.input]
    if image_inputs != [item.name for item in candidate.graph.input]:
        raise ValueError("FP32 and candidate graph-input contract differs")
    outputs = [item.name for item in fp32.graph.output]
    if outputs != [item.name for item in candidate.graph.output] or outputs != [
        item.name for item in d8.graph.output
    ]:
        raise ValueError("six-output contract differs across FP32/B2/D8")
    fp_prefix = extract_model(fp32, image_inputs, cut)
    fp_suffix = extract_model(fp32, cut, outputs)
    b2_prefix = extract_model(candidate, image_inputs, cut)
    b2_suffix = extract_model(candidate, cut, outputs)
    d8_suffix = extract_model(d8, cut, outputs)

    source_index = {node.name: index for index, node in enumerate(fp32.graph.node)}
    leaked_upstream = [
        node.name
        for node in fp_suffix.graph.node
        if node.name in source_index
        and source_index[node.name] in upstream
        and node.op_type != "Constant"
    ]
    leaked_downstream = [
        node.name
        for node in fp_prefix.graph.node
        if node.name in source_index and source_index[node.name] in downstream
    ]
    if leaked_upstream or leaked_downstream:
        raise ValueError(
            f"extracted source halves crossed partition: upstream={leaked_upstream}, "
            f"downstream={leaked_downstream}"
        )
    expected_cut = cut
    if [item.name for item in fp_prefix.graph.output] != expected_cut:
        raise ValueError("FP32 prefix cut output order changed")
    for model in (fp_suffix, b2_suffix, d8_suffix):
        if [item.name for item in model.graph.input] != expected_cut:
            raise ValueError("suffix cut input order changed")

    models = {
        "fp32-prefix": fp_prefix,
        "fp32-suffix": fp_suffix,
        "b2-prefix": b2_prefix,
        "b2-suffix": b2_suffix,
        "b2-d8-suffix": d8_suffix,
    }
    rows: list[dict[str, Any]] = []
    for role, model in models.items():
        path = model_root / spec.frontier / f"{spec.frontier}.{role}.onnx"
        manifest = save_model(model, path)
        rows.append(
            {
                "frontier": spec.frontier,
                "region": spec.region,
                "role": role,
                **manifest,
                "checker": "pass",
                "shape_inference": "pass",
            }
        )
    proof = {
        "frontier": spec.frontier,
        "source_upstream_nodes": len(upstream),
        "source_downstream_nodes": len(downstream),
        "cut_tensor_count": len(cut),
        "hidden_cross_edge_count": 0,
        "graph_input_cross_edge_count": 0,
        "duplicate_cut_name_count": 0,
        "unresolved_shape_count": 0,
        "source_suffix_upstream_leak_count": 0,
        "source_prefix_downstream_leak_count": 0,
        "immutable_constant_duplication_policy": "allowed-not-frontier-state",
        "status": "pass",
    }
    return rows, proof


def parse_specs(raw: list[str]) -> tuple[FrontierSpec, ...]:
    if not raw:
        return COARSE_SPECS
    result: list[FrontierSpec] = []
    for value in raw:
        fields = value.split("|", 3)
        if len(fields) not in (3, 4):
            raise ValueError("--frontier expects ID|REGION|ANCHOR[|MODE]")
        result.append(FrontierSpec(*fields[:3], fields[3] if len(fields) == 4 else "cutoff"))
    if len(result) != len({item.frontier for item in result}):
        raise ValueError("duplicate frontier ID")
    return tuple(result)


def plan(options: argparse.Namespace) -> int:
    if options.model_root.exists() or options.report_dir.exists():
        raise RuntimeError("refusing to reuse model or report output directory")
    options.model_root.mkdir(parents=True)
    options.report_dir.mkdir(parents=True)
    fp32 = inferred_model(options.fp32)
    candidate = inferred_model(options.candidate)
    d8 = inferred_model(options.d8)
    tail = inferred_model(options.tail)
    if [item.name for item in fp32.graph.output] != [item.name for item in tail.graph.input]:
        raise ValueError("FP32 inference/tail boundary contract differs")
    if [item.name for item in candidate.graph.output] != [item.name for item in tail.graph.input]:
        raise ValueError("B2 inference/tail boundary contract differs")

    fp_info = value_info_map(fp32)
    source_consumer_indices = consumer_indices(fp32)
    source_rows: list[dict[str, Any]] = []
    source_consumers = consumers(fp32)
    for index, node in enumerate(fp32.graph.node):
        for output_index, output in enumerate(node.output):
            info = fp_info.get(output)
            source_rows.append(
                {
                    "topological_index": index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "domain": node.domain or "ai.onnx",
                    "producer_output_index": output_index,
                    "output_tensor": output,
                    "shape": shape_text(info) if info else "missing",
                    "dtype": dtype_text(info) if info else "missing",
                    "consumer_count": len(source_consumers.get(output, [])),
                    "consumer_nodes": ";".join(item.name for item in source_consumers.get(output, [])),
                    "architecture_region": region_for_node(node.name),
                    "residual_concat_attention_membership": ";".join(
                        marker
                        for marker in ("residual" if node.op_type == "Add" else "", "concat" if node.op_type == "Concat" else "", "attention" if "/attn/" in node.name else "")
                        if marker
                    ) or "none",
                }
            )
    write_tsv(options.report_dir / "source_graph_map.tsv", source_rows)

    region_rows: list[dict[str, Any]] = []
    for region, description, first, last in REGIONS:
        indices = [
            index
            for index, node in enumerate(fp32.graph.node)
            if (module := module_number(node.name)) is not None and first <= module <= last
        ]
        if not indices:
            raise ValueError(f"architecture region absent: {region}")
        region_rows.append(
            {
                "region": region,
                "description": description,
                "first_module": first,
                "last_module": last,
                "first_topological_index": min(indices),
                "last_topological_index": max(indices),
                "node_count": len(indices),
                "status": "frozen-before-task-metrics",
            }
        )
    write_tsv(options.report_dir / "architecture_region_map.tsv", region_rows)

    specs = parse_specs(options.frontier)
    if len(specs) > options.max_frontiers:
        raise ValueError(f"frontier count {len(specs)} exceeds maximum {options.max_frontiers}")
    correspondence: dict[str, dict[str, Any]] = {}
    frontier_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    proof_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for spec in specs:
        try:
            upstream, downstream = partition_for_spec(fp32, spec)
            cut = complete_cut(fp32, upstream, downstream)
            validate_complete_cut(fp32, upstream, downstream, cut)
            for order, tensor in enumerate(cut):
                if tensor not in fp_info or not static_tensor(fp_info[tensor]):
                    raise ValueError(f"dynamic or unresolved shape: {tensor}")
                mapping = semantic_mapping(fp32, candidate, tensor)
                old = correspondence.get(tensor)
                if old is not None and old != mapping:
                    raise ValueError(f"one source tensor maps inconsistently: {tensor}")
                correspondence[tensor] = mapping
                live_rows.append(
                    {
                        "frontier": spec.frontier,
                        "order": order,
                        "source_tensor": tensor,
                        "candidate_float_tensor": mapping["candidate_float_tensor"],
                        "source_producer": mapping["source_producer"],
                        "shape": mapping["shape"],
                        "dtype": mapping["dtype"],
                        "cross_frontier_consumer_count": sum(
                            1
                            for index in source_consumer_indices[tensor]
                            if index in downstream
                        ),
                        "static_shape_dtype": "pass",
                        "mapping_status": "pass",
                    }
                )
            rows, proof = build_splits(
                fp32,
                candidate,
                d8,
                spec,
                cut,
                upstream,
                downstream,
                options.model_root,
            )
            split_rows.extend(rows)
            proof_rows.append(proof)
            frontier_rows.append(
                {
                    "frontier": spec.frontier,
                    "region": spec.region,
                    "mode": spec.mode,
                    "anchor": spec.anchor,
                    "source_upstream_nodes": len(upstream),
                    "source_downstream_nodes": len(downstream),
                    "cut_tensor_count": len(cut),
                    "cut_tensors": ";".join(cut),
                    "selection_status": "frozen-before-task-metrics",
                    "build_status": "pass",
                }
            )
        except Exception as exc:
            rejections.append(
                {
                    "frontier": spec.frontier,
                    "region": spec.region,
                    "anchor": spec.anchor,
                    "reason": str(exc),
                    "status": "rejected",
                }
            )
    write_tsv(options.report_dir / "coarse_frontier_plan.tsv", frontier_rows or [{"status": "none-accepted"}])
    write_tsv(options.report_dir / "frontier_live_tensor_sets.tsv", live_rows or [{"status": "none-accepted"}])
    write_tsv(
        options.report_dir / "qdq_correspondence.tsv",
        list(correspondence.values()) or [{"status": "no-correspondence"}],
    )
    write_tsv(
        options.report_dir / "mapping_rejections.tsv",
        rejections or [{"frontier": "none", "reason": "none", "status": "no-rejections"}],
    )
    write_tsv(options.report_dir / "split_model_identity.tsv", split_rows or [{"status": "none-built"}])
    (options.report_dir / "frontier_graph_proof.md").write_text(
        "# Complete frontier proof\n\n"
        "Frontiers were frozen before reading new task metrics. For every accepted "
        "source partition, the cut is the exact set of all node outputs produced "
        "upstream and consumed downstream. Initializers remain immutable graph "
        "constants; graph inputs may not cross a frontier. Static shape/dtype, "
        "unique names, unique source-producer/output-index mapping, and direct "
        "source-op -> Q -> DQ provenance are mandatory. Extracted source halves "
        "were checked for partition leakage.\n\n"
        + "\n".join(
            f"- {row['frontier']}: {row['cut_tensor_count']} live tensors, "
            f"{row['source_upstream_nodes']} upstream nodes, status {row['status']}."
            for row in proof_rows
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "fp32": {"path": str(options.fp32), "sha256": sha256(options.fp32)},
        "candidate": {"path": str(options.candidate), "sha256": sha256(options.candidate)},
        "d8": {"path": str(options.d8), "sha256": sha256(options.d8)},
        "tail": {"path": str(options.tail), "sha256": sha256(options.tail)},
        "frontiers": [spec.__dict__ for spec in specs],
        "accepted": [row["frontier"] for row in frontier_rows],
        "rejected": rejections,
    }
    (options.report_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if rejections else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fp32", required=True, type=Path)
    result.add_argument("--candidate", required=True, type=Path)
    result.add_argument("--d8", required=True, type=Path)
    result.add_argument("--tail", required=True, type=Path)
    result.add_argument("--model-root", required=True, type=Path)
    result.add_argument("--report-dir", required=True, type=Path)
    result.add_argument("--frontier", action="append", default=[])
    result.add_argument("--max-frontiers", type=int, default=8)
    return result


def main() -> int:
    return plan(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
