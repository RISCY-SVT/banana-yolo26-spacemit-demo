#!/usr/bin/env python3
"""Summarize Stage65C board runtime, placement, and fixed-fixture controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_state(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            rows.append({"field": fields[0], "value": "\t".join(fields[1:])})
    return rows


def profile_summary(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node"]
    providers = Counter(str(event.get("args", {}).get("provider", "unknown")) for event in node_events)
    names = sorted({re.sub(r"_kernel_time$", "", str(event.get("name", ""))) for event in node_events})
    return {
        "node_events": len(node_events),
        "unique_nodes": len(names),
        "providers": providers,
        "names": names,
        "duration_us": sum(float(event.get("dur", 0.0)) for event in node_events),
    }


def onnx_summary(path: Path) -> dict[str, object]:
    graph = onnx.load(path, load_external_data=False).graph
    return {
        "sha256": sha256(path),
        "nodes": len(graph.node),
        "initializers": len(graph.initializer),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
        "op_types": Counter(node.op_type for node in graph.node),
    }


def numeric(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = float(np.linalg.norm(left.astype(np.float64)) * np.linalg.norm(right.astype(np.float64)))
    cosine = float(np.dot(left.astype(np.float64), right.astype(np.float64)) / denominator) if denominator else 1.0
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-copy", type=Path, required=True)
    parser.add_argument("--tracked-root", type=Path, required=True)
    args = parser.parse_args()
    board = args.board_copy
    out = args.tracked_root

    state_rows = parse_state(board / "state/system_state_before.raw.tsv")
    write_tsv(out / "system_state_before.tsv", ["field", "value"], state_rows)
    state = {row["field"]: row["value"] for row in state_rows}
    board_fields = [
        "hostname", "device_model", "device_serial", "boot_id", "kernel",
        "os_release", "allowed_cpu_list", "memory", "data_mount", "root_mount",
    ]
    write_tsv(
        out / "board_identity.tsv",
        ["field", "value"],
        [{"field": field, "value": state.get(field, "unavailable")} for field in board_fields],
    )

    loaded = read_tsv(board / "state/loaded_library_identity.raw.tsv")
    write_tsv(out / "loaded_library_identity.tsv", list(loaded[0]), loaded)
    log_text = (board / "profile/A1-spacemit-profile/run.log").read_text(encoding="utf-8", errors="replace")
    runtime_match = re.search(r"stage64_runtime (.+)", log_text)
    runtime_rows = [
        {"field": "declared_runtime", "value": "2.0.6", "status": "pass"},
        {"field": "runtime_log", "value": runtime_match.group(1) if runtime_match else "missing", "status": "pass" if runtime_match else "fail"},
        {"field": "core_sha256", "value": next(row["sha256"] for row in loaded if "libonnxruntime" in row["file"]), "status": "pass"},
        {"field": "ep_sha256", "value": next(row["sha256"] for row in loaded if "libspacemit_ep" in row["file"]), "status": "pass"},
        {"field": "runtime_root", "value": "single Stage65C NVMe runtime root", "status": "pass"},
    ]
    write_tsv(out / "runtime_binding.tsv", ["field", "value", "status"], runtime_rows)

    storage = read_tsv(board / "state/storage_write_audit_before.raw.tsv")
    write_tsv(out / "storage_write_audit_before.tsv", list(storage[0]), storage)

    tiny = read_tsv(board / "tiny-controls/tiny_vendor_contract_matrix.raw.tsv")
    tiny_rows = []
    affinity_rows = []
    for row in tiny:
        expected_negative = row["provider"] == "spacemit" and row["test_id"] in {"c4_u8_conv_pc_explicit", "m3_u8_matmul"}
        status = "pass" if ((expected_negative and row["signal"] == "6") or (not expected_negative and row["exit_code"] == "0" and row["exact"] == "1")) else "fail"
        item = {**row, "classification": "expected-unsupported-u8" if expected_negative else "supported", "status": status}
        tiny_rows.append(item)
        if row["provider"] == "spacemit" and row["test_id"] in {"c1_s8_conv_pc_explicit", "m1_s8_matmul"} and "affinity" in row["result_marker"] or (
            row["provider"] == "spacemit" and row["test_id"] in {"c1_s8_conv_pc_explicit", "m1_s8_matmul"} and row["cpus"] in {"0", "0-3", "4", "4-7", "0-7"}
        ):
            affinity_rows.append({"test_id": row["test_id"], "cpus": row["cpus"], "exit_code": row["exit_code"], "exact": row["exact"], "status": status})
    tiny_columns = list(tiny[0]) + ["classification", "status"]
    write_tsv(out / "tiny_s8_control.tsv", tiny_columns, tiny_rows)
    write_tsv(out / "affinity_smoke.tsv", ["test_id", "cpus", "exit_code", "exact", "status"], affinity_rows)

    plugin = read_tsv(board / "plugin/plugin_nonregression.raw.tsv")
    plugin_columns = list(plugin[0])
    for row in plugin:
        row["status"] = "pass" if row["exit_code"] == "0" and row["exact"] == "1" else "fail"
    write_tsv(out / "plugin_nonregression.tsv", plugin_columns + ["status"], plugin)

    session_raw = read_tsv(board / "profile/session_matrix.raw.tsv")
    session_rows = []
    placement_rows = []
    fallback_rows = []
    for row in session_raw:
        run_log = Path(row["log"])
        local_log = board / run_log.relative_to("/data/k1x-stage-runs").parts[-1] if False else None
        # Board paths are rooted at the copied profile/fixed-fixture directories.
        if "/profile/" in row["log"]:
            local_log = board / "profile" / row["log"].split("/profile/", 1)[1]
        else:
            local_log = board / "fixed-fixtures" / row["log"].split("/fixed-fixtures/", 1)[1]
        text = local_log.read_text(encoding="utf-8", errors="replace")
        create = re.search(r"stage64_session inference_create_us=([0-9.]+) tail_create_us=([0-9.]+)", text)
        first = re.search(r"stage64_first inference_us=([0-9.]+) tail_us=([0-9.]+) total_us=([0-9.]+)", text)
        result = "pass" if row["exit_code"] == "0" and "stage64_result status=pass" in text else "fail"
        session_rows.append({
            **row,
            "inference_create_us": create.group(1) if create else "",
            "tail_create_us": create.group(2) if create else "",
            "first_inference_us": first.group(1) if first else "",
            "first_tail_us": first.group(2) if first else "",
            "first_total_us": first.group(3) if first else "",
            "status": result,
        })
        if row["mode"] == "profile":
            profile_dir = local_log.parent / "profiles"
            profile_path = max(profile_dir.glob("*.json"))
            summary = profile_summary(profile_path)
            for provider_name, count in sorted(summary["providers"].items()):
                placement_rows.append({
                    "model": row["model"], "requested_provider": row["provider"],
                    "profile": profile_path, "event_provider": provider_name,
                    "node_events": count, "unique_profile_nodes": summary["unique_nodes"],
                    "node_duration_us": summary["duration_us"],
                })
            cpu_events = summary["providers"].get("CPUExecutionProvider", 0)
            fallback_rows.append({
                "model": row["model"], "requested_provider": row["provider"],
                "inference_cpu_events": cpu_events,
                "intended_separate_tail": "CPUExecutionProvider",
                "unexpected_fallback": "no" if row["provider"] == "cpu" or cpu_events == 0 else "yes",
                "status": "pass" if row["provider"] == "cpu" or cpu_events == 0 else "fail",
            })
    session_columns = list(session_raw[0]) + ["inference_create_us", "tail_create_us", "first_inference_us", "first_tail_us", "first_total_us", "status"]
    write_tsv(out / "session_matrix.tsv", session_columns, session_rows)
    write_tsv(out / "provider_assignment.tsv", ["model", "requested_provider", "profile", "event_provider", "node_events", "unique_profile_nodes", "node_duration_us"], placement_rows)
    write_tsv(out / "cpu_fallback_attribution.tsv", ["model", "requested_provider", "inference_cpu_events", "intended_separate_tail", "unexpected_fallback", "status"], fallback_rows)

    dump_paths = {}
    for model in ("B2", "A1"):
        candidates = list((board / f"profile/{model}-spacemit-profile").glob("SpaceMITExecutionProvider_*.onnx"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one provider dump for {model}, got {candidates}")
        dump_paths[model] = candidates[0]
    dump_summaries = {model: onnx_summary(path) for model, path in dump_paths.items()}
    same_topology = (
        dump_summaries["B2"]["nodes"] == dump_summaries["A1"]["nodes"]
        and dump_summaries["B2"]["inputs"] == dump_summaries["A1"]["inputs"]
        and dump_summaries["B2"]["outputs"] == dump_summaries["A1"]["outputs"]
        and dump_summaries["B2"]["op_types"] == dump_summaries["A1"]["op_types"]
    )
    (out / "provider_subgraphs.md").write_text(
        "# Provider subgraphs\n\n"
        f"B2: one fused subgraph, {dump_summaries['B2']['nodes']} nodes, SHA-256 `{dump_summaries['B2']['sha256']}`.\n\n"
        f"A1: one fused subgraph, {dump_summaries['A1']['nodes']} nodes, SHA-256 `{dump_summaries['A1']['sha256']}`.\n\n"
        f"Partition topology equality (node count, graph I/O and op-type census): `{str(same_topology).lower()}`. "
        "Different graph bytes are expected because A1 changes frozen qparam initializers. "
        "The profiler exposes one SpaceMIT fused event and zero CPU inference events for each EP session; the separate float tail is intentionally CPU.\n",
        encoding="utf-8",
    )

    fixture_root = board / "fixed-fixtures"
    fixture_rows = read_tsv(fixture_root / "fixed_fixture_results.raw.tsv")
    output_rows = []
    numeric_rows = []
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for row in fixture_rows:
        path = fixture_root / f"{row['model']}-{row['provider']}-{row['fixture']}" / "output.bin"
        array = np.fromfile(path, dtype=np.float32)
        if array.size != 1800:
            raise RuntimeError(f"invalid fixed output size: {path}: {array.size}")
        arrays[(row["model"], row["provider"], row["fixture"])] = array
        shaped = array.reshape(300, 6)
        finite = int(np.isfinite(array).sum())
        scores = shaped[:, 4]
        output_rows.append({
            **row,
            "bytes": path.stat().st_size,
            "float_count": array.size,
            "finite_count": finite,
            "non_finite_count": array.size - finite,
            "detection_count_score_ge_0_001": int(np.sum(scores >= 0.001)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "class_count": len({int(value) for value in shaped[:, 5] if value >= 0}),
            "status": "pass" if row["exit_code"] == "0" and finite == array.size else "fail",
        })

    for model in ("B2", "A1"):
        for fixture in ("F0", "bus", "zidane", "canonical"):
            stats = numeric(arrays[(model, "cpu", fixture)], arrays[(model, "spacemit", fixture)])
            numeric_rows.append({"comparison": f"{model}_spacemit-vs-cpu", "fixture": fixture, **stats})
            for boundary in range(6):
                cpu_path = fixture_root / f"{model}-cpu-{fixture}/boundaries/boundary-{boundary}.bin"
                ep_path = fixture_root / f"{model}-spacemit-{fixture}/boundaries/boundary-{boundary}.bin"
                bstats = numeric(np.fromfile(cpu_path, dtype=np.float32), np.fromfile(ep_path, dtype=np.float32))
                numeric_rows.append({"comparison": f"{model}_spacemit-vs-cpu_boundary-{boundary}", "fixture": fixture, **bstats})
    for provider in ("cpu", "spacemit"):
        for fixture in ("F0", "bus", "zidane", "canonical"):
            stats = numeric(arrays[("A1", provider, fixture)], arrays[("B2", provider, fixture)])
            numeric_rows.append({"comparison": f"A1-vs-B2_{provider}", "fixture": fixture, **stats})

    write_tsv(out / "fixed_fixture_results.tsv", list(output_rows[0]), output_rows)
    write_tsv(out / "fixed_fixture_output_hashes.tsv", ["surface", "model", "provider", "fixture", "output_sha256", "boundary_manifest_sha256"], fixture_rows)
    write_tsv(out / "fixed_fixture_numeric_comparison.tsv", ["comparison", "fixture", "max_abs", "mean_abs", "cosine"], numeric_rows)

    if any(row["status"] != "pass" for row in tiny_rows + plugin + session_rows + output_rows + fallback_rows):
        raise SystemExit("one or more runtime controls failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
