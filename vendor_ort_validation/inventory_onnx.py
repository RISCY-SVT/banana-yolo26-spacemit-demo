#!/usr/bin/env python3
"""Inventory model and dumped EP-subgraph structure with exact file identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_names(values: object) -> str:
    return ",".join(value.name for value in values)


def main() -> int:
    options = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(options.models):
        model = onnx.load(path, load_external_data=False)
        operations = Counter(node.op_type for node in model.graph.node)
        rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "ir_version": model.ir_version,
                "opset_imports": ",".join(
                    f"{item.domain or 'ai.onnx'}:{item.version}"
                    for item in model.opset_import
                ),
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "inputs": value_names(model.graph.input),
                "outputs": value_names(model.graph.output),
                "operator_counts": ",".join(
                    f"{name}:{count}" for name, count in sorted(operations.items())
                ),
            }
        )

    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, delimiter="\t", fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
