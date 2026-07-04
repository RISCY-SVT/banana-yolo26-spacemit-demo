# RVV Requant Lookup Report

classification: pass

Candidate: `A2_rvv_f32_lut`

Implementation:

- Vector load corrected int32 accumulators.
- Vector int32 -> f32 conversion.
- Vector multiply by per-channel scale.
- Vector divide by conv output scale.
- Vector convert to int32 code, add zero-point, clamp to `[0,255]`.
- Scalar LUT lookup/write for the small per-channel code vector.

Toolchain:

- Compiler: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- GCC: `14.3.0`
- Flags: `-march=rv64gcv_zvfh -mabi=lp64d`

Disassembly evidence:

- `run_logs/rvv_objdump_excerpt.log`
- Observed RVV instructions include `vsetvli`, `vle32.v`, `vfcvt.f.x.v`, `vfmul.vv`, `vfdiv.vf`, `vfcvt.x.f.v`, `vadd.vx`, `vmax.vx`, `vmin.vx`, and `vse32.v`.

Board correctness:

| CPU | direct Act0/Act1 RVV mismatches | runner Conv2 mismatches |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |

CPU0 benchmark:

| metric | value |
|---|---:|
| selected_subset_total_us | 182420 |
| activation_total_us | 24471.3 |
| activation_share | 13.4148% |
| checksum | 707794080 |
| mismatches | 0 |

This is selected-subset microbench evidence only, not a YOLO26 model FPS claim.
