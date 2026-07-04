# Stage 7 Baseline Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`
previous_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE7-BACKBONE-SUBSET-EXPANSION-001`
baseline_status: pass

## Host Baseline

- Host-native build forced `/usr/bin/gcc` and `/usr/bin/g++`.
- Host CTest before Stage 8 code changes: `19/19` pass.
- Host CTest after Stage 8 changes: `21/21` pass.

## Cross Baseline

- Toolchain: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- Version: SpacemiT GCC `14.3.0`
- Flags: `-march=rv64gcv_zvfh -mabi=lp64d`
- `Y26_K1X_ENABLE_IME=ON`
- Cross build status: pass

## Board Baseline

Board:

- target: `svt@banana`
- kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
- online CPUs: `0-7`
- IME runs: CPU0 only for baseline bench

Initial replay before Stage 8 code changes:

| bucket | us |
|---|---:|
| scalar_total | 1237840 |
| ime_total | 590986 |
| conv0_ime | 70726.9 |
| act0_requant_fallback | 288042 |
| conv1_ime | 65312.6 |
| act1_requant_fallback | 143869 |
| conv2_ime | 21864 |
| activation_total | 431910 |
| activation_share | 73.083% |

Same-board Stage 7 replay after Stage 8 code integration with `scalar_float_reference` mode:

| bucket | us |
|---|---:|
| scalar_total | 1287110 |
| ime_total | 640886 |
| conv0_ime | 71872.9 |
| act0_requant_fallback | 317064 |
| conv1_ime | 68206.8 |
| act1_requant_fallback | 157572 |
| conv2_ime | 24966 |
| activation_total | 474636 |
| activation_share | 74.0593% |

The baseline was reproducible within expected single-run board variance. The Stage 8 comparison uses the same Stage 8 bench binary before/after buckets:

| bucket | us |
|---|---:|
| ime_scalar_float_reference_total | 620735 |
| ime_scalar_float_reference_activation | 465901 |
| ime_int8_lut_total | 350092 |
| ime_int8_lut_activation | 192568 |

No full YOLO26 inference, camera, COCO/mAP, or model FPS measurement was run.
