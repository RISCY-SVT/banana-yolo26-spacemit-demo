#!/usr/bin/env python3
"""Emit the auditable Stage60 future standard-RVV register budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


CONFIGURATIONS = (
    ("A", 256, 32, 32, "no-rename"),
    ("B", 512, 32, 32, "no-rename"),
    ("C", 512, 32, 40, "renamed"),
    ("D", 512, 32, 48, "renamed"),
    ("E", 512, 32, 64, "renamed"),
)

KERNELS = {
    "M8xN16_fused_requant_lut_store": {
        256: (32, 32, "proven register-safe by existing destructive M8 scout"),
        512: (24, 32, "eight LMUL1 accumulators leave a bounded destructive epilogue budget"),
    },
    "M12xN16_fused_requant_lut_store": {
        256: (40, 48, "24-register accumulator file leaves no complete C8 epilogue budget"),
        512: (28, 36, "twelve LMUL1 accumulators plus destructive C8 epilogue fit"),
    },
    "M12xN16_two_iteration_load_ahead": {
        256: (38, 46, "24 accumulator registers plus two seven-register A/B sets exceed v0-v31"),
        512: (26, 34, "twelve LMUL1 accumulators plus two seven-register A/B sets fit"),
    },
    "attention_matmul_c8_epilogue": {
        256: (38, 46, "full compute-to-C8 fusion exceeds the known VLEN256 budget"),
        512: (26, 34, "destructive neighboring-C4 epilogue fits after input registers die"),
    },
    "lut2_indexed_route": {
        256: (7, 15, "e8 data and legal e16,m2 index route use v0-v6"),
        512: (7, 15, "same architectural names; larger VLMAX only changes loop count"),
    },
}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for config, vlen, architectural, physical, rename in CONFIGURATIONS:
        bytes_per_register = vlen // 8
        for kernel, budget in KERNELS.items():
            architectural_peak, physical_demand, reason = budget[vlen]
            no_spill = architectural_peak <= architectural
            physical_headroom = physical - physical_demand if rename == "renamed" else "not-applicable"
            rows.append({
                "configuration": config,
                "kernel": kernel,
                "VLEN_bits": vlen,
                "ELEN_bits": 64,
                "architectural_registers": architectural,
                "physical_registers": physical,
                "rename_policy": rename,
                "architectural_peak_register_names": architectural_peak,
                "physical_destination_demand_if_renamed": physical_demand,
                "architectural_no_spill": int(no_spill),
                "physical_headroom": physical_headroom,
                "architectural_vrf_raw_bytes": architectural * bytes_per_register,
                "physical_vrf_raw_bytes": physical * bytes_per_register,
                "minimum_dot_read_bytes_per_issue": 3 * bytes_per_register,
                "minimum_dot_write_bytes_per_issue": bytes_per_register,
                "assumed_banks": 8 if vlen == 256 else 16,
                "minimum_inflight_vector_operations": 8,
                "fractional_LMUL_required": 1,
                "strong_chaining_bypass_required": 1,
                "one_vector_load_store_pipeline_required": 1,
                "reason": reason,
            })
    write_tsv(args.output, rows)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
