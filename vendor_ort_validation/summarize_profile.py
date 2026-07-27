#!/usr/bin/env python3
"""Summarize ONNX Runtime JSON profile events without inferring placement."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def profile_label(path: Path) -> str:
    name = path.name
    if "__" in name:
        name = name.split("__", 1)[0]
    return name


def node_events(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: profile root is not an array")
    result: list[dict[str, Any]] = []
    for event in raw:
        if event.get("cat") != "Node" or event.get("ph") != "X":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        if not isinstance(provider, str) or not provider:
            continue
        result.append(event)
    return result


def main() -> int:
    options = parse_args()
    options.events.parent.mkdir(parents=True, exist_ok=True)
    options.summary.parent.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for profile in options.profiles:
        events = node_events(profile)
        label = profile_label(profile)
        provider_counts: Counter[str] = Counter()
        provider_duration: defaultdict[str, int] = defaultdict(int)
        op_counts: Counter[tuple[str, str]] = Counter()
        op_duration: defaultdict[tuple[str, str], int] = defaultdict(int)
        total_duration = 0

        for event in events:
            args = event["args"]
            provider = str(args["provider"])
            op_name = str(args.get("op_name", "unknown"))
            duration_us = int(event.get("dur", 0))
            total_duration += duration_us
            provider_counts[provider] += 1
            provider_duration[provider] += duration_us
            op_counts[(provider, op_name)] += 1
            op_duration[(provider, op_name)] += duration_us
            event_rows.append(
                {
                    "profile": label,
                    "profile_file": profile.name,
                    "event_name": event.get("name", ""),
                    "node_index": args.get("node_index", ""),
                    "op_name": op_name,
                    "provider": provider,
                    "duration_us": duration_us,
                    "execution_count": 1,
                }
            )

        for (provider, op_name), count in sorted(op_counts.items()):
            duration_us = op_duration[(provider, op_name)]
            summary_rows.append(
                {
                    "profile": label,
                    "provider": provider,
                    "op_name": op_name,
                    "profile_event_count": count,
                    "duration_us": duration_us,
                    "duration_fraction": (
                        f"{duration_us / total_duration:.9f}" if total_duration else "unknown"
                    ),
                    "provider_event_count": provider_counts[provider],
                    "provider_duration_us": provider_duration[provider],
                    "provider_duration_fraction": (
                        f"{provider_duration[provider] / total_duration:.9f}"
                        if total_duration
                        else "unknown"
                    ),
                    "scope_note": (
                        "an EP subgraph profile event can contain many original ONNX nodes"
                    ),
                }
            )

    event_fields = [
        "profile",
        "profile_file",
        "event_name",
        "node_index",
        "op_name",
        "provider",
        "duration_us",
        "execution_count",
    ]
    with options.events.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, delimiter="\t", fieldnames=event_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(event_rows)

    summary_fields = [
        "profile",
        "provider",
        "op_name",
        "profile_event_count",
        "duration_us",
        "duration_fraction",
        "provider_event_count",
        "provider_duration_us",
        "provider_duration_fraction",
        "scope_note",
    ]
    with options.summary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, delimiter="\t", fieldnames=summary_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
