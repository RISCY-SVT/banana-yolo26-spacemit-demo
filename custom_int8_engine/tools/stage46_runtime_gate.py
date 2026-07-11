#!/usr/bin/env python3
"""Generate Stage46 plugin fixtures and compact runtime/package evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def archive_audit(args: argparse.Namespace) -> None:
    archive = Path(args.archive).resolve()
    rows: list[dict[str, Any]] = []
    unsafe: list[str] = []
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            unsafe_path = name.is_absolute() or ".." in name.parts
            unsafe_link = False
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                unsafe_link = link.is_absolute() or ".." in link.parts
            if unsafe_path or unsafe_link:
                unsafe.append(member.name)
            rows.append(
                {
                    "path": member.name,
                    "type": "symlink" if member.issym() else "hardlink" if member.islnk() else "dir" if member.isdir() else "file",
                    "size": member.size,
                    "mode": oct(member.mode),
                    "link_target": member.linkname if member.issym() or member.islnk() else "",
                    "unsafe": int(unsafe_path or unsafe_link),
                }
            )
    write_tsv(Path(args.output), rows, ["path", "type", "size", "mode", "link_target", "unsafe"])
    result = {
        "archive": str(archive),
        "sha256": sha256_file(archive),
        "size": archive.stat().st_size,
        "members": len(rows),
        "unsafe_members": unsafe,
    }
    print(json.dumps(result, indent=2))
    if unsafe:
        raise SystemExit(2)


def package_manifest(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if path.is_symlink():
            kind = "symlink"
            target = os.readlink(path)
            digest = ""
        elif path.is_file():
            kind = "file"
            target = ""
            digest = sha256_file(path)
        elif path.is_dir():
            kind = "dir"
            target = ""
            digest = ""
        else:
            kind = "other"
            target = ""
            digest = ""
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "kind": kind,
                "size": info.st_size,
                "mode": stat.filemode(info.st_mode),
                "sha256": digest,
                "link_target": target,
            }
        )
    write_tsv(Path(args.output), rows, ["path", "kind", "size", "mode", "sha256", "link_target"])
    print(json.dumps({"root": str(root), "entries": len(rows)}, indent=2))


def package_diff(args: argparse.Namespace) -> None:
    def load(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline="") as stream:
            return {row["path"]: row for row in csv.DictReader(stream, delimiter="\t")}

    before = load(Path(args.before).resolve())
    after = load(Path(args.after).resolve())
    rows: list[dict[str, Any]] = []
    for path in sorted(before.keys() | after.keys()):
        lhs = before.get(path)
        rhs = after.get(path)
        if lhs is None:
            status = "added"
        elif rhs is None:
            status = "removed"
        elif lhs["kind"] != rhs["kind"]:
            status = "kind_changed"
        elif lhs["sha256"] != rhs["sha256"] or lhs["link_target"] != rhs["link_target"]:
            status = "content_changed"
        elif lhs["mode"] != rhs["mode"]:
            status = "mode_changed"
        else:
            status = "unchanged"
        rows.append(
            {
                "path": path,
                "status": status,
                "before_kind": lhs["kind"] if lhs else "",
                "after_kind": rhs["kind"] if rhs else "",
                "before_size": lhs["size"] if lhs else "",
                "after_size": rhs["size"] if rhs else "",
                "before_sha256": lhs["sha256"] if lhs else "",
                "after_sha256": rhs["sha256"] if rhs else "",
                "before_link_target": lhs["link_target"] if lhs else "",
                "after_link_target": rhs["link_target"] if rhs else "",
            }
        )
    fields = [
        "path", "status", "before_kind", "after_kind", "before_size", "after_size",
        "before_sha256", "after_sha256", "before_link_target", "after_link_target",
    ]
    write_tsv(Path(args.output), rows, fields)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"rows": len(rows), "status_counts": counts}, indent=2, sort_keys=True))


def make_plugin_fixture(args: argparse.Namespace) -> None:
    import onnx
    from onnx import TensorProto, helper

    output = Path(args.model).resolve()
    input_path = Path(args.input).resolve()
    expected_path = Path(args.expected).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(args.elements, dtype=np.uint32).astype(np.uint8)
    expected = np.bitwise_xor(values, np.uint8(0x5A))
    input_path.write_bytes(values.tobytes())
    expected_path.write_bytes(expected.tobytes())
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["pre"], name="stage46/pre_identity"),
            helper.make_node(
                "Stage46U8Xor", ["pre"], ["xor"], name="stage46/plugin_u8_xor", domain="spacemit.custom"
            ),
            helper.make_node("Identity", ["xor"], ["output"], name="stage46/post_identity"),
        ],
        "stage46_plugin_partition_graph",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, args.elements])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, args.elements])],
    )
    model = helper.make_model(
        graph,
        producer_name="banana-yolo26-stage46",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("spacemit.custom", 1)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, output)
    print(
        json.dumps(
            {
                "model": str(output),
                "model_sha256": sha256_file(output),
                "input": str(input_path),
                "input_sha256": sha256_file(input_path),
                "expected": str(expected_path),
                "expected_sha256": sha256_file(expected_path),
                "elements": args.elements,
            },
            indent=2,
        )
    )


def compare_raw(args: argparse.Namespace) -> None:
    actual = Path(args.actual).read_bytes()
    expected = Path(args.expected).read_bytes()
    if len(actual) != len(expected):
        print(json.dumps({"structural_valid": False, "actual_bytes": len(actual), "expected_bytes": len(expected)}))
        raise SystemExit(2)
    lhs = np.frombuffer(actual, dtype=np.uint8).astype(np.int16)
    rhs = np.frombuffer(expected, dtype=np.uint8).astype(np.int16)
    diff = lhs - rhs
    result = {
        "structural_valid": True,
        "elements": int(lhs.size),
        "mismatches": int(np.count_nonzero(diff)),
        "max_abs_diff": int(np.max(np.abs(diff))) if diff.size else 0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "byte_equal": actual == expected,
    }
    print(json.dumps(result, indent=2))
    if actual != expected:
        raise SystemExit(1)


def run_host_fixtures(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    model = Path(args.model).resolve()
    fixture_root = Path(args.fixture_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = levels[args.opt_level]
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.intra_op_num_threads = args.threads
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.type != "tensor(float)" or any(not isinstance(dim, int) or dim < 0 for dim in input_meta.shape):
        raise RuntimeError(f"expected concrete float input, got {input_meta.type} {input_meta.shape}")
    expected_elements = int(np.prod(input_meta.shape, dtype=np.int64))
    rows: list[dict[str, Any]] = []
    for fixture_dir in sorted(path for path in fixture_root.glob("F*") if path.is_dir()):
        input_path = fixture_dir / "images.bin"
        values = np.fromfile(input_path, dtype=np.float32)
        if values.size != expected_elements:
            raise RuntimeError(
                f"fixture {fixture_dir.name} element mismatch: {values.size} != {expected_elements}"
            )
        tensor = values.reshape(input_meta.shape)
        output = np.asarray(session.run([output_meta.name], {input_meta.name: tensor})[0])
        output_path = output_dir / f"{fixture_dir.name}.bin"
        output.tofile(output_path)
        rows.append(
            {
                "fixture": fixture_dir.name,
                "model_sha256": sha256_file(model),
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_path),
                "output_dtype": str(output.dtype),
                "output_shape": "x".join(str(dim) for dim in output.shape),
                "output_min": float(np.min(output)),
                "output_max": float(np.max(output)),
                "output_mean": float(np.mean(output, dtype=np.float64)),
                "output_nonfinite": int(np.count_nonzero(~np.isfinite(output))),
                "runtime_version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "opt_level": args.opt_level,
                "threads": args.threads,
            }
        )
    fields = list(rows[0]) if rows else ["fixture"]
    write_tsv(Path(args.manifest), rows, fields)
    print(json.dumps({"fixtures": len(rows), "manifest": str(Path(args.manifest).resolve())}, indent=2))


def compare_fixture_outputs(args: argparse.Namespace) -> None:
    reference_dir = Path(args.reference_dir).resolve()
    actual_dir = Path(args.actual_dir).resolve()
    rows: list[dict[str, Any]] = []
    for fixture in args.fixtures.split(","):
        fixture = fixture.strip()
        if not fixture:
            continue
        reference_path = reference_dir / f"{fixture}.bin"
        actual_path = actual_dir / f"{args.actual_prefix}{fixture}.bin"
        reference = np.fromfile(reference_path, dtype=np.float32)
        actual = np.fromfile(actual_path, dtype=np.float32)
        structural_valid = reference.shape == actual.shape
        if structural_valid:
            difference = actual.astype(np.float64) - reference.astype(np.float64)
            finite = np.isfinite(actual) & np.isfinite(reference)
            exact = np.equal(actual, reference) | (np.isnan(actual) & np.isnan(reference))
            rows.append(
                {
                    "fixture": fixture,
                    "reference_path": str(reference_path),
                    "actual_path": str(actual_path),
                    "reference_sha256": sha256_file(reference_path),
                    "actual_sha256": sha256_file(actual_path),
                    "elements": reference.size,
                    "structural_valid": 1,
                    "mismatches": int(np.count_nonzero(~exact)),
                    "mismatch_ratio": float(np.mean(~exact)),
                    "max_abs_diff": float(np.max(np.abs(difference[finite]))) if np.any(finite) else "nan",
                    "mean_abs_diff": float(np.mean(np.abs(difference[finite]))) if np.any(finite) else "nan",
                    "rmse": float(np.sqrt(np.mean(np.square(difference[finite])))) if np.any(finite) else "nan",
                    "reference_nonfinite": int(np.count_nonzero(~np.isfinite(reference))),
                    "actual_nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
                }
            )
        else:
            rows.append(
                {
                    "fixture": fixture,
                    "reference_path": str(reference_path),
                    "actual_path": str(actual_path),
                    "reference_sha256": sha256_file(reference_path),
                    "actual_sha256": sha256_file(actual_path),
                    "elements": f"{reference.size}/{actual.size}",
                    "structural_valid": 0,
                    "mismatches": "",
                    "mismatch_ratio": "",
                    "max_abs_diff": "",
                    "mean_abs_diff": "",
                    "rmse": "",
                    "reference_nonfinite": int(np.count_nonzero(~np.isfinite(reference))),
                    "actual_nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
                }
            )
    fields = list(rows[0]) if rows else ["fixture"]
    write_tsv(Path(args.output), rows, fields)
    print(json.dumps({"rows": len(rows), "output": str(Path(args.output).resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("archive-audit")
    command.add_argument("--archive", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=archive_audit)

    command = subparsers.add_parser("package-manifest")
    command.add_argument("--root", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=package_manifest)

    command = subparsers.add_parser("package-diff")
    command.add_argument("--before", required=True)
    command.add_argument("--after", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=package_diff)

    command = subparsers.add_parser("make-plugin-fixture")
    command.add_argument("--model", required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--expected", required=True)
    command.add_argument("--elements", type=int, default=4096)
    command.set_defaults(func=make_plugin_fixture)

    command = subparsers.add_parser("compare-raw")
    command.add_argument("--actual", required=True)
    command.add_argument("--expected", required=True)
    command.set_defaults(func=compare_raw)

    command = subparsers.add_parser("run-host-fixtures")
    command.add_argument("--model", required=True)
    command.add_argument("--fixture-root", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--opt-level", choices=["disable", "basic", "extended", "all"], default="disable")
    command.add_argument("--threads", type=int, default=1)
    command.set_defaults(func=run_host_fixtures)

    command = subparsers.add_parser("compare-fixture-outputs")
    command.add_argument("--reference-dir", required=True)
    command.add_argument("--actual-dir", required=True)
    command.add_argument("--actual-prefix", required=True)
    command.add_argument("--fixtures", default="F0,F5,F6,F7")
    command.add_argument("--output", required=True)
    command.set_defaults(func=compare_fixture_outputs)

    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
