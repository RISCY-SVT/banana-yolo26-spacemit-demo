# First Divergence Report

## Method

Host ORT 1.27.0 and board ORT 1.20.2+spacemit used identical model and input bytes under `ORT_DISABLE_ALL`, sequential execution, one intra/inter thread, disabled memory pattern/arena, no appended vendor EP. Graph-derived cuts were compared from model.0 through output0.

## First divergent boundary

The float output immediately before the first output quantizer is byte-identical:

```text
tensor: /model.0/conv/Conv_output_0
shape: 1x16x320x320 float32
host raw SHA256: 29dd483b205eb6754084553b022d5abc0cda8e9fafb51c06884978079d0c1645
board raw SHA256: 29dd483b205eb6754084553b022d5abc0cda8e9fafb51c06884978079d0c1645
```

The next boundary is the first divergence:

```text
tensor: /model.0/conv/Conv_output_0_QuantizeLinear_Output
shape: 1x16x320x320 uint8
mismatches: 5 / 1638400
max_abs_diff: 1
first_mismatch_index: 178797
host-minus-board histogram: -1:5, 0:1638395
```

Verbose logs prove that all six nodes in this cut, including Conv and QuantizeLinear, were assigned to CPUExecutionProvider. Because the float Conv output is exact and the following uint8 output differs, the earliest bounded difference is the vendor/upstream QuantizeLinear execution path. The exact internal cause may be rounding, vector kernel, or another implementation detail; Stage42 does not claim a source-level root cause without vendor source evidence.

## Propagation

The mismatch grows downstream: model4 input has 1498 differing codes (max 4), model4 output has 25994 (max 6), model16 quantized output has 114490 (max 53), and the pre-head model22 boundary has 25844 (max 34). Under the full diagnostic cut, output0 differs in 1711/1800 floats with max absolute difference `637.887696743`.

Complete graph metadata and statistics are in `boundary_tensor_manifest.tsv` and `boundary_comparison.tsv`. Cross-runtime detection-output diagnostics are in `cross_runtime_output0_diagnostic.md`; they are not mAP or accuracy evidence.
