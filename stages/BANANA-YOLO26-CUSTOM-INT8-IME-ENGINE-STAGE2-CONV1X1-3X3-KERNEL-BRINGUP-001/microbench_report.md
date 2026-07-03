# Microbench Report

All numbers are kernel-level measurements. They are not YOLO26 FPS, full-image speed, camera speed, COCO mAP, or production readiness evidence.

## Vmadot Hotpath

Command:

```bash
taskset -c 0 ./bench_vmadot_hotpath 20000 5
```

| path | packing included | mean ns/call | stddev ns/call | speedup vs scalar |
| --- | --- | ---: | ---: | ---: |
| scalar | no | 251.982 | 0.875 | 1.000 |
| public cached wrapper | no | 95.453 | 0.320 | 2.640 |
| unsafe cluster0 | no | 60.178 | 0.372 | 4.187 |

## Conv Kernels

Command:

```bash
taskset -c 0-3 ./bench_conv_kernels 200 5
```

| case | shape | packing included | scalar mean | IME mean | speedup vs scalar | mismatches |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Conv1x1 MMT4D tile core | `4x4x8` | no | 253.361 ns | 38.759 ns | 6.537 | 0 |
| Conv3x3 MMT4D tile core | `4x4x8` | no | 251.471 ns | 39.268 ns | 6.404 | 0 |
| Conv1x1 wrapper | `8x8x16 -> OC16` | yes | 85.702 us | 94.621 us | 0.906 | 0 |
| Conv3x3 wrapper | `8x8x16 -> OC16` | yes | 669.155 us | 1834.478 us | 0.365 | 0 |

Interpretation: the IME tile core is healthy; current simple packing/im2col wrappers are not performance-ready.
