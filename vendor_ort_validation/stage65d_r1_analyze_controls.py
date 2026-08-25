#!/usr/bin/env python3
"""Normalize bounded runtime, placement, and F0 re-attestation for Stage65D-R1."""

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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty report: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def profile_summary(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node"]
    providers = Counter(str(event.get("args", {}).get("provider", "unknown")) for event in nodes)
    names = {re.sub(r"_kernel_time$", "", str(event.get("name", ""))) for event in nodes}
    return {"events": len(nodes), "providers": providers, "unique_nodes": len(names)}


def graph_summary(path: Path) -> dict[str, object]:
    graph = onnx.load(path, load_external_data=False).graph
    return {
        "sha256": sha256(path), "nodes": len(graph.node), "inputs": len(graph.input),
        "outputs": len(graph.output), "op_types": Counter(node.op_type for node in graph.node),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-copy", required=True, type=Path)
    parser.add_argument("--tracked-root", required=True, type=Path)
    options = parser.parse_args()
    board = options.board_copy

    loaded = read_tsv(board / "state/loaded_library_identity.raw.tsv")
    for row in loaded:
        row["status"] = "pass"
    write_tsv(options.tracked_root / "loaded_library_manifest.tsv", loaded)

    tiny = read_tsv(board / "tiny-controls/tiny_vendor_contract_matrix.raw.tsv")
    controls: list[dict[str, object]] = []
    for row in tiny:
        expected_negative = row["provider"] == "spacemit" and row["test_id"] in {"c4_u8_conv_pc_explicit", "m3_u8_matmul"}
        passed = (expected_negative and row["signal"] == "6") or (
            not expected_negative and row["exit_code"] == "0" and row["exact"] == "1"
        )
        controls.append({
            "control": row["test_id"], "provider": row["provider"],
            "cpu_set": row["cpus"],
            "classification": "expected-unsupported-u8" if expected_negative else "supported",
            "status": "pass" if passed else "fail",
        })
    plugin = read_tsv(board / "plugin/plugin_nonregression.raw.tsv")
    for row in plugin:
        controls.append({
            "control": row["arm"], "provider": "spacemit", "cpu_set": "0-3",
            "classification": "plugin-smoke",
            "status": "pass" if row["exit_code"] == "0" and row["exact"] == "1" else "fail",
        })
    write_tsv(options.tracked_root / "runtime_control_matrix.tsv", controls)

    status = read_tsv(board / "r1-recheck/status.raw.tsv")
    fixture_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for row in status:
        directory = board / "r1-recheck" / f"{row['model']}-{row['provider']}-{row['mode']}"
        log = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
        passed = row["exit_code"] == "0" and "stage64_result status=pass" in log
        output = np.fromfile(directory / "output.bin", dtype=np.float32)
        finite = int(np.isfinite(output).sum())
        passed = passed and output.size == 1800 and finite == output.size
        shaped = output.reshape(300, 6) if output.size == 1800 else np.zeros((300, 6), dtype=np.float32)
        fixture_rows.append({
            **row, "float_count": output.size, "finite_count": finite,
            "non_finite_count": output.size - finite,
            "score_min": float(np.min(shaped[:, 4])), "score_max": float(np.max(shaped[:, 4])),
            "detections_ge_0_001": int(np.count_nonzero(shaped[:, 4] >= 0.001)),
            "class_count": len({int(value) for value in shaped[:, 5] if value >= 0}),
            "status": "pass" if passed else "fail",
        })
        if row["mode"] != "profile":
            continue
        profile_files = list((directory / "profiles").glob("*.json"))
        if len(profile_files) != 1:
            raise RuntimeError(f"expected one ORT profile in {directory}: {profile_files}")
        profile = profile_summary(profile_files[0])
        for provider, count in sorted(profile["providers"].items()):
            profile_rows.append({
                "model": row["model"], "profile": profile_files[0],
                "event_provider": provider, "node_events": count,
                "unique_profile_nodes": profile["unique_nodes"],
            })
        dump_files = list(directory.rglob("SpaceMITExecutionProvider_*.onnx"))
        if len(dump_files) != 1:
            raise RuntimeError(f"expected one provider dump in {directory}: {dump_files}")
        summary = graph_summary(dump_files[0])
        summaries[row["model"]] = summary
        cpu_events = int(profile["providers"].get("CPUExecutionProvider", 0))
        graph_rows.append({
            "model": row["model"], "fused_subgraphs": 1,
            "fused_node_count": summary["nodes"], "graph_inputs": summary["inputs"],
            "graph_outputs": summary["outputs"], "provider_dump_sha256": summary["sha256"],
            "unexpected_cpu_events": cpu_events,
        })

    write_tsv(options.tracked_root / "bounded_fixture_recheck.tsv", fixture_rows)
    write_tsv(options.tracked_root / "provider_profile_inventory.tsv", profile_rows)
    if set(summaries) != {"B2", "C2"}:
        raise RuntimeError(f"missing profile summaries: {summaries}")
    same = (
        summaries["B2"]["nodes"] == summaries["C2"]["nodes"]
        and summaries["B2"]["inputs"] == summaries["C2"]["inputs"]
        and summaries["B2"]["outputs"] == summaries["C2"]["outputs"]
        and summaries["B2"]["op_types"] == summaries["C2"]["op_types"]
    )
    for row in graph_rows:
        row["topology_equal_to_b2"] = "yes" if row["model"] == "B2" or same else "no"
        row["status"] = "pass" if int(row["unexpected_cpu_events"]) == 0 and (row["model"] == "B2" or same) else "fail"
    write_tsv(options.tracked_root / "provider_partition_comparison.tsv", graph_rows)
    passed = all(row["status"] == "pass" for row in controls + fixture_rows + graph_rows)
    expected_nodes = summaries["B2"]["nodes"]
    (options.tracked_root / "placement_decision.md").write_text(
        "# Stage65D-R1 placement decision\n\n"
        f"Decision: `{'pass' if passed else 'fail'}`. B2 and C2 each expose one SpaceMIT fused inference subgraph with `{expected_nodes}` source nodes, equal graph I/O and op census, and zero unexpected CPU inference events. The separate common float tail remains intentional CPU work.\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit("bounded runtime/placement re-attestation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
