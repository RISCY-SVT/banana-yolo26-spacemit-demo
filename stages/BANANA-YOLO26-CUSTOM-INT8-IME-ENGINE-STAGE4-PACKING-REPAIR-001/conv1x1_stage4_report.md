# Conv1x1 Stage 4 Report

Selected real node: `/model.2/cv1/conv/Conv`

Shape: `160x160x32->32`, 1x1 stride1 pad0.

Correctness:

- host CTest: pass
- board Stage 3 real fixture: pass
- board Stage 4 persistent M-major: pass
- board Stage 4 persistent N-major: pass

Board microbench, `bench_stage4_packing 5`, CPU0:

| metric | us |
|---|---:|
| scalar | `112047` |
| old on-the-fly IME wrapper | `137374` |
| Stage 3 baseline prepacked | `46649.6` |
| Stage 4 persistent M-major | `21843.2` |
| Stage 4 persistent N-major | `65855.4` |
| prepack object create | `44.759` |
| workspace create | `0.825` |
| Stage 4 packA probe | `6260.54` |
| correction | `2649.96` |

Decision: Conv1x1 Stage 4 path is accepted for first-block integration experiments.
