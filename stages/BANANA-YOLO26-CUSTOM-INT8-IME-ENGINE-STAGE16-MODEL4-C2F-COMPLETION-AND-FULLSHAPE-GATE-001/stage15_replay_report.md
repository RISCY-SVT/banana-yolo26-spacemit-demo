# Stage 15 Replay Report

subset: `candidate_I_model4_split_first_branch`
replay_status: `pass`
shape_class: `compact`

Stage 15 was replayed before and after Stage 16 changes. Host CTest passed and board CPU0/1/2/3 correctness passed with `mismatches=0`.

CPU0 compact replay microbench:

| candidate | total_us | conv_us | activation_requant_us | merge_us | mismatches |
|---|---:|---:|---:|---:|---:|
| `stage15_IME_A2_rvv_f32_lut` | 159.38 | 102.767 | 31.0707 | 12.3623 | 0 |

Evidence logs:

- `run_logs/host_ctest_stage16_validation.log`
- `run_logs/board_final_cpu0_bench_stage15.log`
- `run_logs/board_final_cpu0_stage15.log`
- `run_logs/board_final_cpu1_stage15.log`
- `run_logs/board_final_cpu2_stage15.log`
- `run_logs/board_final_cpu3_stage15.log`
