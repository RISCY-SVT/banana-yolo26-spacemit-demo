# Block Microbench Report

Scope: selected block only, not YOLO26 FPS, not full model speed, not full-image pipeline speed, not camera speed, not COCO/mAP, and not production readiness.

Selected block:

- `block0_conv_only`
- `/model.0/conv/Conv`
- shape: `640x640x3 -> 320x320x16`
- kernel: `3x3`
- stride: `2`
- padding: `1`

Board command:

- `taskset -c 0 ./bench_stage5_first_block 3`

Board result:

| metric | us | status | notes |
|---|---:|---:|---|
| `scalar_total` | `463480` | `0` | corrected int32 output |
| `ime_prepack_one_time` | `5457.43` | `0` | prepare/destroy measured separately |
| `ime_packA_probe` | `38912.7` | `0` | local packA probe for selected block |
| `ime_total_packing_included` | `71932.7` | `0` | persistent prepack/workspace, corrected output |
| `ime_compute_plus_correction_residual` | `33020` | n/a | `ime_total - packA_probe` |

Checksums:

- scalar checksum over 3 runs: `20167891605`
- IME checksum over 3 runs: `20167891605`

Memory:

- prepacked bytes: `576`
- workspace bytes: `128`
- raw int32 scratch bytes for full block: `6553600`

Decision:

- `ime_total_packing_included` is faster than `scalar_total` for the selected first block.
- Stage 6 can move to a multi-block backbone subset after review.
