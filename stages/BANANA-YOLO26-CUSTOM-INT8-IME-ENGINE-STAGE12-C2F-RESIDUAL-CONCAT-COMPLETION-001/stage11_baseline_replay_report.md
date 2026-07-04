# Stage 11 Baseline Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE12-C2F-RESIDUAL-CONCAT-COMPLETION-001`
baseline_subset: `candidate_F_model2_m0_cv1_act_cv2_conv`
board_cpu: `CPU0`
command: `taskset -c 0 ./bench_stage11_branch_block 3`
raw_log: `run_logs/board_stage11_replay_bench_cpu0.log`

## Correctness

Board CPU0/1/2/3 replay passed.

- `test_stage10_rvv_rounding_control`: `mismatches=0`, `after_frm` preserved for ambient `frm=0..4`
- `test_stage11_branch_block_runner`: `branch0_act_mismatches=0`, `branch1_mismatches=0`

## CPU0 Timing Replay

| path | total_us | mismatches | activation_share_pct | conv_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|
| `stage11_scalar_reference` | `1313820` | `0` | `20.8853` | `78.9793` | `0.0906631` |
| `stage11_scalar_A2_rvv_f32_lut` | `1060840` | `0` | `3.77667` | `96.0931` | `0.074781` |
| `stage11_IME_A2_rvv_f32_lut` | `263171` | `0` | `14.9207` | `84.5192` | `0.298207` |

## Decision

Stage 11 replay is clean. Stage 12 work starts from a valid baseline.
