# Stage 12 Baseline Replay Report

start_head: `cae5301afc10a1ff2138335932d4939e3db64fc2`
host_ctest: pass `29/29` before Stage 13 code, pass `30/30` after Stage 13 test registration
cross_build: pass
board_cpu0_to_cpu3_correctness: pass

## Binary Hashes

Final RISC-V binaries:

| binary | sha256 |
|---|---|
| `test_stage12_c2f_block_runner` | `ee72f811c255d34d8a461491be7b645f59e3e72c45cc11e3f4969e4ef71d7bad` |
| `test_stage13_merge_dataflow` | `cacd0e1ed5b897746f25edb1cb502c7a59adb394c3a1a85cb1aaef09dea6d7cf` |
| `bench_stage12_c2f_block` | `8fea0a53cfc001e9d297e3f782eb3eb8036ebcb9e3ceb6ef17fce3573f8b4fd3` |
| `bench_stage13_merge_dataflow` | `a7ad78ee94657db2ebd40374542359b7dc4222a73d63c8bec54690b4d8b583aa` |

## Stage 12 Replay

Final CPU0 `bench_stage12_c2f_block 3`:

| path | total_us | conv_us | activation_requant_us | split_us | add_us | concat_us | post_concat_qdq_us | pack_layout_us | correction_us | model2_cv2_conv_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage12_IME_A2_rvv_f32_lut` | `608164` | `284093` | `96232.2` | `133954` | `3358.98` | `5429.07` | `82770` | `945.182` | `3455.69` | `53467` | `0` |

The total has normal board drift relative to the Stage 12 final run. Correctness
replayed with `mismatches=0`.

## Double-Counting Finding

Stage 12 final report showed `pack_layout_share_pct=22.3855`. After repairing
bucket definitions, replay reports `pack_layout_share_pct=0.156006`. The prior
number included split/materialization cost in pack/layout and should not be used
as a true Conv pack-layout cost.
