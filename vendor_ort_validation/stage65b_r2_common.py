#!/usr/bin/env python3
"""Shared bounded I/O helpers for Stage65B-R2 host evidence."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


MAX_CSV_FIELD_SIZE = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    previous = csv.field_size_limit()
    csv.field_size_limit(MAX_CSV_FIELD_SIZE)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))
    except csv.Error as exc:
        if "field larger than field limit" in str(exc):
            raise ValueError(
                f"TSV field exceeds the {MAX_CSV_FIELD_SIZE}-character limit: {path}"
            ) from exc
        raise
    finally:
        csv.field_size_limit(previous)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
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
