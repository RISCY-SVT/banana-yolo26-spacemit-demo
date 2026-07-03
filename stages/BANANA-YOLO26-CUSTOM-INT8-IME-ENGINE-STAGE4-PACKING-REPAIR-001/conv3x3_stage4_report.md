# Conv3x3 Stage 4 Report

Selected real node: `/model.2/m.0/cv1/conv/Conv`

Shape: `160x160x16->8`, 3x3 stride1 pad1.

Correctness:

- host CTest: pass
- board Stage 3 real fixture: pass
- board Stage 4 persistent M-major: pass
- board Stage 4 persistent N-major: pass

Board microbench, `bench_stage4_packing 5`, CPU0:

| metric | us |
|---|---:|
| scalar | `147333` |
| old on-the-fly IME wrapper | `391010` |
| Stage 3 baseline prepacked | `149121` |
| Stage 4 persistent M-major | `37097.9` |
| Stage 4 persistent N-major | `65852.3` |
| prepack object create | `51.4756` |
| workspace create | `1.225` |
| Stage 4 packA probe | `24951.5` |
| correction | `658.726` |

The raw benchmark checksum differs from scalar for Conv3x3 because the repaired real-node path uses quantized padding value `input_storage_zero_point_s8`, while the old synthetic scalar benchmark path uses raw signed zero padding. The real selected-node correctness tests compare after the documented correction boundary and pass with zero mismatches.

Decision: Conv3x3 Stage 4 path is accepted for first-block integration experiments, with a residual note that packA remains the largest component.
