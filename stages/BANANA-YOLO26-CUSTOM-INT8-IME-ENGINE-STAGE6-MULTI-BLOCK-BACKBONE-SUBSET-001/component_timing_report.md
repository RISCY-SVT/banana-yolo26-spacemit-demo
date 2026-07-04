# Component Timing Report

Board command:

```text
taskset -c 0 ./bench_stage6_multiblock 3
```

## Stage 6 Selected Subset

| component | scalar mean us | IME mean us | notes |
| --- | ---: | ---: | --- |
| Conv0 + correction | `446010` | `67775.3` | Existing Stage 4/5 prepacked M-major Conv path. |
| Activation/requant | `286179` | `286942` | Shared scalar float fallback; measured separately. |
| Conv1 + correction | `276642` | `63886.5` | Existing Stage 4 prepacked Conv path on Conv1. |
| total | `1009980` | `419769` | Selected subset only. |

## PackA Probe

| component | mean us | notes |
| --- | ---: | --- |
| Conv0 packA probe | `38341.8` | Probe of A packing work for Conv0 shape. |
| Conv1 packA probe | `31799.9` | Probe of A packing work for Conv1 shape after activation handoff. |

The benchmark keeps kernel/block measurements separate from any full model or image-pipeline timing.

