#!/usr/bin/env python3
"""Normalize Stage65E graph census and read-only runtime capability probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto


STAGE64_DIRECT_E2E = Path(
    "/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/stages/"
    "BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-"
    "YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001/direct_e2e_diagnostic.tsv"
)
STAGE64_DIRECT_E2E_SHA256 = (
    "064b9b89086f0a9c3bdcc84420a9f084f3c9a72fc13c4e3e8e0e9f4765818422"
)
STAGE64_CAUSAL_DECISION = STAGE64_DIRECT_E2E.with_name("causal_decision.md")
STAGE64_CAUSAL_DECISION_SHA256 = (
    "15c892bc2529900dcc91a591187110943ce4483fbb3e85b04e5388abc0ab1565"
)


STATS_RE = re.compile(
    r"stage46_stats metric=wall mean_us=(?P<mean>[0-9.]+).*?"
    r"median_us=(?P<median>[0-9.]+).*?p95_us=(?P<p95>[0-9.]+)"
)
SESSION_RE = re.compile(r"stage46_session status=created create_us=(?P<create>[0-9.]+)")
FIRST_RE = re.compile(r"stage46_first_run first_run_us=(?P<first>[0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def tensor_contract(value: onnx.ValueInfoProto) -> tuple[str, str]:
    tensor = value.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor.elem_type)
    dimensions = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return dtype, "x".join(dimensions)


def graph_census(model_name: str, role: str, path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    op_types = Counter(node.op_type for node in graph.node)
    rows: list[dict[str, object]] = [{
        "model": model_name,
        "graph_role": role,
        "record_type": "summary",
        "name": graph.name or path.name,
        "op_type": "all",
        "count": len(graph.node),
        "dtype": "n/a",
        "shape": "n/a",
        "sha256": sha256(path),
    }]
    for op_type, count in sorted(op_types.items()):
        rows.append({
            "model": model_name,
            "graph_role": role,
            "record_type": "operator-census",
            "name": op_type,
            "op_type": op_type,
            "count": count,
            "dtype": "n/a",
            "shape": "n/a",
            "sha256": sha256(path),
        })
    for direction, values in (("input", graph.input), ("output", graph.output)):
        for value in values:
            dtype, shape = tensor_contract(value)
            rows.append({
                "model": model_name,
                "graph_role": role,
                "record_type": direction,
                "name": value.name,
                "op_type": "tensor",
                "count": 1,
                "dtype": dtype,
                "shape": shape,
                "sha256": sha256(path),
            })
    return rows, {
        "nodes": len(graph.node),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
        "sha256": sha256(path),
        "qdq": op_types["QuantizeLinear"] + op_types["DequantizeLinear"],
        "op_types": op_types,
        "output_contract": [(value.name, *tensor_contract(value)) for value in graph.output],
    }


def parse_perf_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stats_match = STATS_RE.search(text)
    session_match = SESSION_RE.search(text)
    first_match = FIRST_RE.search(text)
    return {
        "session_create_us": session_match.group("create") if session_match else "unavailable",
        "first_run_us": first_match.group("first") if first_match else "unavailable",
        "mean_us": stats_match.group("mean") if stats_match else "unavailable",
        "median_us": stats_match.group("median") if stats_match else "unavailable",
        "p95_us": stats_match.group("p95") if stats_match else "unavailable",
    }


def profile_providers(path: Path) -> tuple[int, int]:
    events = json.loads(path.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node"]
    spacemit = sum(
        event.get("args", {}).get("provider") == "SpaceMITExecutionProvider"
        for event in node_events
    )
    cpu = sum(
        event.get("args", {}).get("provider") == "CPUExecutionProvider"
        for event in node_events
    )
    return spacemit, cpu


def profile_partition_summary(
    path: Path,
    source_summary: dict[str, object],
    accepted_partition: dict[str, str],
    accepted_profile: Path,
) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node"]
    ep_events = [
        event
        for event in node_events
        if event.get("args", {}).get("provider") == "SpaceMITExecutionProvider"
    ]
    op_names = sorted({event.get("args", {}).get("op_name", "") for event in ep_events})
    if len(op_names) != 1 or not op_names[0]:
        raise RuntimeError(f"profile does not prove exactly one SpaceMIT partition: {path}")
    accepted_events = json.loads(accepted_profile.read_text(encoding="utf-8"))
    accepted_op_names = sorted({
        event.get("args", {}).get("op_name", "")
        for event in accepted_events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "SpaceMITExecutionProvider"
    })
    if op_names != accepted_op_names:
        raise RuntimeError(f"SpaceMIT partition identity differs from accepted profile: {path}")
    if (
        accepted_partition["status"] != "pass"
        or accepted_partition["fused_subgraphs"] != "1"
        or accepted_partition["fused_node_count"] != "925"
        or accepted_partition["graph_inputs"] != "1"
        or accepted_partition["graph_outputs"] != "6"
        or accepted_partition["unexpected_cpu_events"] != "0"
    ):
        raise RuntimeError("accepted 925-node partition contract is invalid")
    input_contracts = {
        json.dumps(event.get("args", {}).get("input_type_shape", []), sort_keys=True)
        for event in ep_events
    }
    output_contracts = {
        json.dumps(event.get("args", {}).get("output_type_shape", []), sort_keys=True)
        for event in ep_events
    }
    if len(input_contracts) != 1 or len(output_contracts) != 1:
        raise RuntimeError(f"SpaceMIT partition contract changed within profile: {path}")
    payload = {
        "provider": "SpaceMITExecutionProvider",
        "op_names": op_names,
        "accepted_provider_dump_sha256": accepted_partition["provider_dump_sha256"],
        "source_graph_sha256": source_summary["sha256"],
        "source_partition_nodes": int(accepted_partition["fused_node_count"]),
        "partition_inputs": int(accepted_partition["graph_inputs"]),
        "partition_outputs": int(accepted_partition["graph_outputs"]),
        "input_type_shape": json.loads(next(iter(input_contracts))),
        "output_type_shape": json.loads(next(iter(output_contracts))),
    }
    identity = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "nodes": int(accepted_partition["fused_node_count"]),
        "inputs": int(accepted_partition["graph_inputs"]),
        "outputs": int(accepted_partition["graph_outputs"]),
        "fused_subgraphs": 1,
        "sha256": identity,
        "topology_sha256": identity,
    }


def graph_summary(path: Path) -> dict[str, object]:
    graph = onnx.load(path, load_external_data=False).graph
    topology_payload = {
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "nodes": [
            {
                "name": node.name,
                "domain": node.domain,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for node in graph.node
        ],
    }
    return {
        "nodes": len(graph.node),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
        "op_types": Counter(node.op_type for node in graph.node),
        "sha256": sha256(path),
        "topology_sha256": hashlib.sha256(
            json.dumps(
                topology_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def accepted_fixture_signature(root: Path, model: str) -> list[tuple[int, str]]:
    fixture = root / f"{model}-spacemit-fixture"
    boundary_dir = fixture / "boundaries"
    files = sorted(boundary_dir.glob("boundary-*.bin"))
    expected_files = [boundary_dir / f"boundary-{index}.bin" for index in range(6)]
    if files != expected_files:
        raise RuntimeError(f"accepted {model} fixture does not contain exactly six ordered boundaries")
    signature = [(path.stat().st_size, sha256(path)) for path in expected_files]
    declared = []
    for line in (fixture / "boundary-sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, _path = line.split(maxsplit=1)
        declared.append(digest)
    if declared != [digest for _bytes, digest in signature]:
        raise RuntimeError(f"accepted {model} fixture boundary manifest drift")
    return signature


def probe_boundary_signature(
    path: Path,
    expected_names: list[str],
) -> list[tuple[int, str]]:
    rows = read_tsv(path)
    if [int(row["index"]) for row in rows] != list(range(6)):
        raise RuntimeError(f"probe boundary ordering drift: {path}")
    if [row["name"] for row in rows] != expected_names:
        raise RuntimeError(f"probe boundary-name contract drift: {path}")
    return [(int(row["bytes"]), row["sha256"]) for row in rows]


def status_markdown(title: str, state: str, paragraphs: list[str]) -> str:
    body = "\n\n".join(paragraphs)
    return f"# {title}\n\nStatus: `{state}`.\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    parser.add_argument("--frozen-identity", required=True, type=Path)
    parser.add_argument("--performance-ratios", required=True, type=Path)
    parser.add_argument("--tail-timing", required=True, type=Path)
    parser.add_argument("--accepted-fixture-root", required=True, type=Path)
    parser.add_argument("--xslim-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    options = parser.parse_args()

    frozen = read_tsv(options.frozen_identity)
    paths = {
        (row["surface"], row["kind"]): Path(row["path"])
        for row in frozen
    }
    b2 = paths[("B2", "inference")]
    c2 = paths[("C2", "inference")]
    tail = paths[("common", "tail")]

    census_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for model_name, model_path in (("B2", b2), ("C2", c2)):
        rows, summary = graph_census(model_name, "inference", model_path)
        census_rows.extend(rows)
        summaries[model_name] = summary
    tail_rows, tail_summary = graph_census("common", "cpu-fp32-tail", tail)
    census_rows.extend(tail_rows)
    if summaries["B2"]["output_contract"] != summaries["C2"]["output_contract"]:
        raise RuntimeError("B2/C2 graph-output contract mismatch")
    tail_model = onnx.load(tail, load_external_data=False)
    tail_inputs = [(value.name, *tensor_contract(value)) for value in tail_model.graph.input]
    if summaries["B2"]["output_contract"] != tail_inputs:
        raise RuntimeError("six-output host/device boundary does not match the common tail")
    if summaries["B2"]["qdq"] != 812 or summaries["C2"]["qdq"] != 812:
        raise RuntimeError("frozen Q/DQ census changed")
    fixture_signatures = {
        model: accepted_fixture_signature(options.accepted_fixture_root, model)
        for model in ("B2", "C2")
    }
    for name, dtype, shape in tail_inputs:
        census_rows.append({
            "model": "B2/C2-common",
            "graph_role": "SpaceMIT-to-CPU-tail-boundary",
            "record_type": "host-device-boundary",
            "name": name,
            "op_type": "tensor",
            "count": 1,
            "dtype": dtype,
            "shape": shape,
            "sha256": sha256(tail),
        })
    write_tsv(options.tracked_root / "main_graph_and_tail_census.tsv", census_rows)

    accepted_tail_median: dict[str, float] = {}
    for row in read_tsv(options.tail_timing):
        if row["metric"] == "tail" and row["model"] in {"B2", "C2"}:
            accepted_tail_median[row["model"]] = float(row["median_us"])
    if set(accepted_tail_median) != {"B2", "C2"}:
        raise RuntimeError("accepted B2/C2 tail timing reference is incomplete")

    accepted_partitions = {
        row["model"]: row
        for row in read_tsv(options.tracked_root / "provider_partition_comparison.tsv")
        if row["model"] in {"B2", "C2"}
    }
    accepted_profiles = {
        row["model"]: Path(row["profile"])
        for row in read_tsv(options.tracked_root / "provider_profile_inventory.tsv")
        if row["model"] in {"B2", "C2"}
    }
    if set(accepted_partitions) != {"B2", "C2"} or set(accepted_profiles) != {"B2", "C2"}:
        raise RuntimeError("accepted provider-partition evidence is incomplete")
    if any(not path.is_file() for path in accepted_profiles.values()):
        raise RuntimeError("accepted provider profile is unavailable")

    optimization_status = read_tsv(options.raw_root / "optimization-status.raw.tsv")
    optimization_rows: list[dict[str, object]] = []
    disable_two_stage_medians: dict[str, float] = {}
    level_two_stage_medians: dict[tuple[str, str], float] = {}
    for status in optimization_status:
        model, level = status["model"], status["opt_level"]
        directory = options.raw_root / f"optimization-{model}-{level}"
        profiles = list((directory / "profile").glob("*.json"))
        dumps = list((directory / "provider-dumps").rglob("SpaceMITExecutionProvider_*.onnx"))
        if int(status["exit_code"]) != 0 or len(profiles) != 1 or len(dumps) > 1:
            optimization_rows.append({
                "model": model,
                "opt_level": level,
                "exit_code": status["exit_code"],
                "session_create_us": "unavailable",
                "first_run_us": "unavailable",
                "mean_us": "unavailable",
                "median_us": "unavailable",
                "p95_us": "unavailable",
                "output_sha256": status["output_sha256"],
                "profile_output_sha256": status["profile_output_sha256"],
                "profile_output_equal_to_unprofiled": "unresolved",
                "output_equal_to_disable": "unresolved",
                "tail_median_us_reference": accepted_tail_median[model],
                "projected_two_stage_median_us": "unavailable",
                "two_stage_measurement_kind": "not-available-probe-failed",
                "boundary_count": status["boundary_count"],
                "boundary_manifest_sha256": status["boundary_manifest_sha256"],
                "six_boundary_equal_to_disable": "unresolved",
                "six_boundary_equal_to_accepted_fixture": "unresolved",
                "fused_subgraphs": len(dumps),
                "partition_evidence": "unavailable",
                "partition_nodes": "unavailable",
                "partition_inputs": "unavailable",
                "partition_outputs": "unavailable",
                "unexpected_cpu_events": "unavailable",
                "partition_sha256": "unavailable",
                "partition_topology_sha256": "unavailable",
                "partition_topology_equal_to_disable": "unresolved",
                "decision": "reject-probe-failed",
            })
            continue
        spacemit_events, cpu_events = profile_providers(profiles[0])
        timing = parse_perf_log(directory / "run.log")
        if dumps:
            partition = graph_summary(dumps[0])
            partition_evidence = "provider-subgraph-dump"
        else:
            partition = profile_partition_summary(
                profiles[0],
                summaries[model],
                accepted_partitions[model],
                accepted_profiles[model],
            )
            partition_evidence = (
                "current-single-EP-profile-matched-accepted-925-node-partition-contract"
            )
        output_hash = status["output_sha256"]
        profile_output_hash = status["profile_output_sha256"]
        boundary_count = int(status["boundary_count"])
        boundary_manifest_hash = status["boundary_manifest_sha256"]
        median = float(timing["median_us"])
        projected_two_stage_median = median + accepted_tail_median[model]
        level_two_stage_medians[(model, level)] = projected_two_stage_median
        if level == "disable":
            disable_two_stage_medians[model] = projected_two_stage_median
        if partition["nodes"] != 925 or cpu_events != 0 or spacemit_events == 0:
            probe_decision = "reject-placement-drift"
        elif boundary_count != 6 or boundary_manifest_hash == "missing":
            probe_decision = "reject-six-boundary-capture"
        else:
            probe_decision = "pass"
        optimization_rows.append({
            "model": model,
            "opt_level": level,
            "exit_code": status["exit_code"],
            **timing,
            "output_sha256": output_hash,
            "profile_output_sha256": profile_output_hash,
            "profile_output_equal_to_unprofiled": "pending",
            "output_equal_to_disable": "pending",
            "tail_median_us_reference": accepted_tail_median[model],
            "projected_two_stage_median_us": projected_two_stage_median,
            "two_stage_measurement_kind": "unprofiled-main-plus-exact-accepted-tail-reference",
            "boundary_count": boundary_count,
            "boundary_manifest_sha256": boundary_manifest_hash,
            "six_boundary_equal_to_disable": "pending",
            "six_boundary_equal_to_accepted_fixture": "pending",
            "fused_subgraphs": partition.get("fused_subgraphs", 1),
            "partition_evidence": partition_evidence,
            "partition_nodes": partition["nodes"],
            "partition_inputs": partition["inputs"],
            "partition_outputs": partition["outputs"],
            "unexpected_cpu_events": cpu_events,
            "partition_sha256": partition["sha256"],
            "partition_topology_sha256": partition["topology_sha256"],
            "partition_topology_equal_to_disable": "pending",
            "decision": probe_decision,
        })
    for row in optimization_rows:
        if row["decision"] == "reject-probe-failed":
            continue
        disable_hash = next(
            item["output_sha256"]
            for item in optimization_rows
            if item["model"] == row["model"] and item["opt_level"] == "disable"
        )
        equal = row["output_sha256"] == disable_hash
        row["output_equal_to_disable"] = "yes" if equal else "no"
        if not equal:
            row["decision"] = "reject-output-drift"
        profile_equal = row["profile_output_sha256"] == row["output_sha256"]
        row["profile_output_equal_to_unprofiled"] = "yes" if profile_equal else "no"
        if not profile_equal:
            row["decision"] = "reject-profile-output-drift"
        disable_boundary_hash = next(
            item["boundary_manifest_sha256"]
            for item in optimization_rows
            if item["model"] == row["model"] and item["opt_level"] == "disable"
        )
        boundaries_equal = (
            row["boundary_count"] == 6
            and row["boundary_manifest_sha256"] == disable_boundary_hash
        )
        row["six_boundary_equal_to_disable"] = "yes" if boundaries_equal else "no"
        if not boundaries_equal:
            row["decision"] = "reject-six-boundary-drift"
        probe_signature = probe_boundary_signature(
            options.raw_root
            / f"optimization-{row['model']}-{row['opt_level']}"
            / "boundary-output-manifest.tsv",
            [name for name, _dtype, _shape in summaries[str(row["model"])]["output_contract"]],
        )
        fixture_equal = probe_signature == fixture_signatures[str(row["model"])]
        row["six_boundary_equal_to_accepted_fixture"] = "yes" if fixture_equal else "no"
        if not fixture_equal:
            row["decision"] = "reject-accepted-fixture-output-drift"
        disable_topology_hash = next(
            item["partition_topology_sha256"]
            for item in optimization_rows
            if item["model"] == row["model"] and item["opt_level"] == "disable"
        )
        topology_equal = row["partition_topology_sha256"] == disable_topology_hash
        row["partition_topology_equal_to_disable"] = "yes" if topology_equal else "no"
        if not topology_equal:
            row["decision"] = "reject-partition-topology-drift"
    write_tsv(options.tracked_root / "ort_optimization_matrix.tsv", optimization_rows)

    capability = read_tsv(options.raw_root / "capability-status.raw.tsv")
    by_probe: dict[str, list[dict[str, str]]] = {}
    for row in capability:
        by_probe.setdefault(row["probe"], []).append(row)
    offline_rows = by_probe.get("offline-optimized", [])
    offline_readback_rows = by_probe.get("offline-optimized-readback", [])
    offline_created = bool(offline_rows) and all(
        row["exit_code"] == "0"
        and int(row["artifact_bytes"]) > 0
        and row["artifact_sha256"] != "missing"
        for row in offline_rows
    )
    offline_supported = (
        offline_created
        and len(offline_readback_rows) == 2
        and all(row["exit_code"] == "0" for row in offline_readback_rows)
    )
    offline_state = (
        "supported-create-and-readback-capability-only"
        if offline_supported
        else "creation-only-readback-failed"
        if offline_created
        else "unsupported"
    )
    options.tracked_root.joinpath("offline_optimization_capability.md").write_text(
        status_markdown(
            "Offline optimization capability",
            offline_state,
            [
                "The shipped target `onnxruntime_perf_test` was invoked on the physical K1X with the exact SpaceMIT provider. "
                + ("It emitted and successfully read back non-empty optimized ONNX artifacts for both frozen models." if offline_supported else "It did not complete valid create-and-readback for both frozen models."),
                "Artifacts remain raw diagnostic bytes and do not replace B2 or C2. This capability alone does not prove a startup or steady-state benefit.",
            ],
        ),
        encoding="utf-8",
    )

    baseline_rows = by_probe.get("iobinding-baseline", [])
    input_rows = by_probe.get("iobinding-input", [])
    input_binding_supported = bool(input_rows) and all(row["exit_code"] == "0" for row in input_rows)
    baseline_supported = bool(baseline_rows) and all(row["exit_code"] == "0" for row in baseline_rows)
    iobinding_state = "input-binding-supported" if input_binding_supported and baseline_supported else "unsupported"
    options.tracked_root.joinpath("iobinding_capability.md").write_text(
        status_markdown(
            "I/O Binding capability",
            iobinding_state,
            [
                "The result is based on the shipped target perf-test API, not generic ORT documentation.",
                "Only input pre-binding was exercised. Output device pre-allocation and a device-resident handoff into the separate CPU FP32 tail were not established; the accepted tail requires six host-readable float tensors.",
            ],
        ),
        encoding="utf-8",
    )

    ep_rows = by_probe.get("ep-context", [])
    ep_readback_rows = by_probe.get("ep-context-readback", [])
    ep_created = bool(ep_rows) and all(
        row["exit_code"] == "0"
        and int(row["artifact_bytes"]) > 0
        and row["artifact_sha256"] != "missing"
        for row in ep_rows
    )
    ep_supported = (
        ep_created
        and len(ep_readback_rows) == 2
        and all(row["exit_code"] == "0" for row in ep_readback_rows)
    )
    ep_state = (
        "supported-create-and-readback-capability-only"
        if ep_supported
        else "creation-only-readback-failed"
        if ep_created
        else "unsupported"
    )
    options.tracked_root.joinpath("ep_context_capability.md").write_text(
        status_markdown(
            "EPContext capability",
            ep_state,
            [
                "The shipped tool accepted the compile flow, emitted artifacts, and read them back successfully for both models." if ep_supported else "The shipped tool did not complete the compile-and-readback flow with valid artifacts for both models.",
                "Any emitted context remains raw, is not accepted model evidence, and has no proven portability or startup benefit until a separately authorized load/readback benchmark succeeds.",
            ],
        ),
        encoding="utf-8",
    )

    plugin_header = options.runtime_root / "spacemit-ort.riscv64.2.0.6/include/spacemit_ort_plugin.h"
    plugin_doc = options.runtime_root / "spacemit-ort.riscv64.2.0.6/plugin/PLUGIN.md"
    plugin_text = plugin_header.read_text(encoding="utf-8", errors="replace") + plugin_doc.read_text(encoding="utf-8", errors="replace")
    plugin_api = "AddOperator" in plugin_text and "AddDispatch" in plugin_text
    options.tracked_root.joinpath("plugin_tail_capability.md").write_text(
        status_markdown(
            "Plugin and tail capability",
            "structurally-possible-not-implemented" if plugin_api else "unsupported-by-audited-api",
            [
                "The shipped plugin contract exposes both full custom-operator registration (`AddOperator`) and dispatch overlay registration (`AddDispatch`)." if plugin_api else "The audited shipped headers and plugin guide do not expose both required registration tracks.",
                f"The accepted CPU tail has `{tail_summary['nodes']}` nodes, six FP32 inputs, and one `1x300x6` FP32 output. A CPU custom op or external C++/RVV stage is structurally conceivable. SpaceMIT ownership, exact numerical equivalence, fused placement, and speed are not proven. No source, custom op, or model was created.",
            ],
        ),
        encoding="utf-8",
    )

    yolo_source = options.xslim_root / "src/xslim/onnxslim_pass/yolo_decode.py"
    yolo_text = yolo_source.read_text(encoding="utf-8", errors="strict")
    yolo_present = "YoloDecodePatternMatcher" in yolo_text and "FusionYoloDecode" in yolo_text
    if sha256(STAGE64_DIRECT_E2E) != STAGE64_DIRECT_E2E_SHA256:
        raise RuntimeError("Stage64 direct-E2E evidence identity drift")
    direct_e2e = next(
        row
        for row in read_tsv(STAGE64_DIRECT_E2E)
        if row["lane"] == "XSLIM_VENDOR_REF_9A33"
    )
    collapse_bound = (
        direct_e2e["status"] == "host-semantic-failure"
        and direct_e2e["host_images"] == "100"
        and direct_e2e["host_passes"] == "0"
        and direct_e2e["score_collapses"] == "100"
    )
    if not collapse_bound:
        raise RuntimeError("Stage64 direct-E2E score-collapse contract drift")
    if sha256(STAGE64_CAUSAL_DECISION) != STAGE64_CAUSAL_DECISION_SHA256:
        raise RuntimeError("Stage64 direct-E2E causal-decision identity drift")
    causal_text = STAGE64_CAUSAL_DECISION.read_text(encoding="utf-8")
    if "all 100 host holdout\nimages have finite `1x300x6` outputs but zero nonzero scores" not in causal_text:
        raise RuntimeError("Stage64 direct-E2E finite-output statement drift")
    options.tracked_root.joinpath("xslim_yolodecode_status.md").write_text(
        status_markdown(
            "XSlim YoloDecode status",
            "source-present-frozen-split-does-not-use-it",
            [
                f"Current XSlim source contains the YoloDecode fusion implementation: `{'yes' if yolo_present else 'no'}`. The frozen B2/C2 inference graphs intentionally stop at six head boundaries and contain no `YoloDecode` node.",
                f"Stage64's exact `direct_e2e_diagnostic.tsv` (SHA-256 `{STAGE64_DIRECT_E2E_SHA256}`) binds the repaired vendor-reference direct-E2E lane to 100 holdout images, zero passes, and 100 score collapses. The exact causal decision (SHA-256 `{STAGE64_CAUSAL_DECISION_SHA256}`) binds those cases to finite `1x300x6` outputs with zero nonzero scores. The defect remains unreconciled, so direct-E2E generation is not justified by this audit.",
                "A future exact candidate would have to preserve the current source/model/qparams, reproduce the 34-node tail exactly, retain finite noncollapsed task output, and re-prove provider placement and COCO accuracy.",
            ],
        ),
        encoding="utf-8",
    )

    tail_share: dict[str, float] = {}
    tail_timing = read_tsv(options.tail_timing)
    for model in ("B2", "C2"):
        tail_median = float(next(row["median_us"] for row in tail_timing if row["model"] == model and row["metric"] == "tail"))
        total_median = float(next(row["median_us"] for row in tail_timing if row["model"] == model and row["metric"] == "two_stage"))
        tail_share[model] = tail_median / total_median

    opportunity_rows: list[dict[str, object]] = []
    for model in ("B2", "C2"):
        opportunity_rows.append({
            "opportunity": "exact-tail-implementation-upper-bound",
            "model": model,
            "measured_value": tail_share[model],
            "threshold": "tail share >= 0.05; projected exact gain >= 0.02",
            "placement_output_status": "unchanged-frozen-model; implementation-not-run",
            "classification": "needs-implementation-benchmark" if tail_share[model] >= 0.05 else "low-roi",
            "evidence": "matched ABBA median tail/two-stage share",
        })
    performance_ratios = read_tsv(options.performance_ratios)
    two_stage_noise_floor = float(next(
        row["comparison_noise_floor"]
        for row in performance_ratios
        if row["metric"] == "two_stage" and row["statistic"] == "median"
    ))
    common_level_gain: dict[str, float] = {}
    for level in ("disable", "basic", "extended", "all"):
        gains = []
        for model in ("B2", "C2"):
            baseline = disable_two_stage_medians.get(model)
            candidate = level_two_stage_medians.get((model, level))
            if baseline is None or candidate is None:
                gains = []
                break
            gains.append(1.0 - candidate / baseline)
        if gains:
            common_level_gain[level] = min(gains)
    best_level, best_steady_gain = max(
        common_level_gain.items(), key=lambda item: item[1], default=("unavailable", 0.0)
    )
    opportunity_rows.extend([
        {
            "opportunity": "ORT-optimization-level",
            "model": "B2/C2",
            "measured_value": best_steady_gain,
            "threshold": "steady gain >= 0.02 beyond noise with exact placement/output",
            "placement_output_status": "see ort_optimization_matrix.tsv",
            "classification": "steady-state-opportunity" if best_steady_gain >= max(0.02, two_stage_noise_floor) else "low-roi",
            "evidence": f"best common level={best_level}; 100-run unprofiled main-partition timing plus exact accepted tail reference; empirical two-stage noise floor={two_stage_noise_floor}",
        },
        {
            "opportunity": "offline-optimized-model",
            "model": "B2/C2",
            "measured_value": "capability-only",
            "threshold": "operationally material startup reduction",
            "placement_output_status": offline_state,
            "classification": "startup-only-unquantified" if offline_supported else "unsupported",
            "evidence": "target perf-test compile probe",
        },
        {
            "opportunity": "EPContext",
            "model": "B2/C2",
            "measured_value": "capability-only",
            "threshold": "operationally material startup reduction",
            "placement_output_status": ep_state,
            "classification": "startup-only-unquantified" if ep_supported else "unsupported",
            "evidence": "target perf-test compile probe",
        },
        {
            "opportunity": "I/O-Binding",
            "model": "B2/C2",
            "measured_value": iobinding_state,
            "threshold": "exact output/placement and >= 0.02 two-stage gain",
            "placement_output_status": "CPU-tail requires host-readable six-output boundary",
            "classification": "no-proven-two-stage-opportunity",
            "evidence": "target perf-test input-binding probe only",
        },
    ])
    for row in offline_rows:
        opportunity_rows.append({
            "opportunity": "offline-optimized-artifact-identity",
            "model": row["model"],
            "measured_value": f"bytes={row['artifact_bytes']};sha256={row['artifact_sha256']}",
            "threshold": "non-empty identity-bound diagnostic artifact",
            "placement_output_status": offline_state,
            "classification": "raw-diagnostic-only",
            "evidence": f"log_sha256={row['log_sha256']}",
        })
    for row in ep_rows:
        opportunity_rows.append({
            "opportunity": "EPContext-artifact-identity",
            "model": row["model"],
            "measured_value": f"bytes={row['artifact_bytes']};sha256={row['artifact_sha256']}",
            "threshold": "non-empty identity-bound diagnostic artifact",
            "placement_output_status": ep_state,
            "classification": "raw-diagnostic-only",
            "evidence": f"log_sha256={row['log_sha256']}",
        })
    for probe in (
        "offline-optimized-readback",
        "iobinding-baseline",
        "iobinding-input",
        "ep-context-readback",
        "ep-device-list",
    ):
        for row in by_probe.get(probe, []):
            opportunity_rows.append({
                "opportunity": f"{probe}-probe",
                "model": row["model"],
                "measured_value": f"exit_code={row['exit_code']}",
                "threshold": "capability readback only",
                "placement_output_status": "not-an-accepted-model-surface",
                "classification": "capability-only" if row["exit_code"] == "0" else "unsupported",
                "evidence": f"log_sha256={row['log_sha256']}",
            })
    write_tsv(options.tracked_root / "fusion_opportunity_ledger.tsv", opportunity_rows)

    # A measured tail share is an upper bound, not a projected implementation gain.
    # With no exact-tail implementation benchmark and no >=2% accepted ORT gain, the
    # prompt's opening threshold is not satisfied.
    placement_output_pass = all(row["decision"] == "pass" for row in optimization_rows)
    opt_qualified = (
        best_steady_gain >= max(0.02, two_stage_noise_floor)
        and placement_output_pass
    )
    stage65f_justified = opt_qualified
    decision = "stage65f-justified" if stage65f_justified else "not-justified-low-roi-or-unquantified"
    options.tracked_root.joinpath("fusion_feasibility_decision.md").write_text(
        status_markdown(
            "Fusion feasibility decision",
            decision,
            [
                f"Measured median CPU-tail shares are B2 `{tail_share['B2']:.6f}` and C2 `{tail_share['C2']:.6f}` of two-stage latency. They establish an upper bound, not a projected exact-tail gain; no tail implementation benchmark is authorized in this Stage.",
                f"The best common B2/C2 read-only projected two-stage ORT optimization-level gain versus `ORT_DISABLE_ALL` is `{best_steady_gain:.6f}` at `{best_level}`; the matched two-stage noise floor is `{two_stage_noise_floor:.6f}`. Main-partition timing is unprofiled; the exact accepted per-model tail median is added as an invariant reference only after six-boundary output identity passes. A >=2% option is accepted only when the gain also exceeds that floor and output plus 925-node placement remain exact.",
                "Offline optimization, I/O Binding, EPContext, plugin-tail, and YoloDecode results are capability evidence only. No accepted model, runtime default, or source was changed.",
            ],
        ),
        encoding="utf-8",
    )
    charter = (
        "# Stage65F charter draft\n\n"
        + ("Status: `justified-not-authorized`.\n\nA later Stage may benchmark the exact proven runtime option under matched placement/output controls.\n" if stage65f_justified else "Status: `not-justified-not-opened`.\n\nNo measured and fully qualified >=2% steady two-stage opportunity, no operationally quantified startup requirement, and no evidence-backed exact-tail projection jointly satisfy the Stage65E opening rule.\n")
        + "\nThis file is a feasibility disposition, not authorization, implementation, promotion, or a later-stage prompt.\n"
    )
    options.tracked_root.joinpath("STAGE65F_CHARTER_DRAFT.md").write_text(charter, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
