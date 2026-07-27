# Draft issue #1 update: SpacemiT ORT 2.0.6 on K1X

This is a draft only. Stage63 did not post or modify the issue.

## Package identity

- Official asset: `spacemit-ort.riscv64.2.0.6.tar.gz`
- Asset SHA-256: `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6`
- Asset size: `15002263` bytes
- ORT core: `1.24.2+spacemit.a1`, embedded build `9bb02204b`
- SpacemiT EP package version: `2.0.6`
- `libonnxruntime.so.1` is byte-identical to 2.0.5.
- `libspacemit_ep.so.2` changed:
  - 2.0.5: `3927b51f79f8d2142ff98708183aa9b24b47d6941533499035193a630042a41d`
  - 2.0.6: `dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae`

The public source tags `2.0.5` and `2.0.6` both resolve to
`61e7fc2319cd16aa5487fd1155dc15c5390c8a90`. These results therefore describe
the official binary release asset; they do not establish source provenance for
the changed provider binary.

## Results

| Surface | 2.0.6 result | Classification |
|---|---|---|
| Q/DQ Conv without `kernel_shape` | EP assigned; exact | unchanged positive control |
| Q/DQ Conv with `kernel_shape=[3,3]` | clip-minmax error, then abort | unchanged |
| minimal `QLinearConv` | SIGILL / exit 132 | unchanged |
| minimal `QLinearMatMul` | SIGILL / exit 132 | unchanged |
| official plugin sample link/load | builds; `ldd -r`, `dlopen`, ABI query and init pass | fixed |
| independent uint8 plugin execution | assigned; exact; 1,011 observed dispatches | fixed |
| official Track2 plugin graph | 31,892 / 32,768 values differ from CPU | new sample correctness defect |
| full YOLO26 Q/DQ INT8 EP | session creation aborts at quantized Conv | unchanged |

The no-`kernel_shape` Q/DQ Conv is a positive control, not a 2.0.6 fix.
For `QLinearConv`, the captured SIGILL PC is in unchanged
`libonnxruntime.so.1`; the instruction word is `0xe204082b`, which the accepted
objdump renders as `.insn`. No instruction semantic is inferred.

## Full-model impact

FP32 and FP16 full models create and execute SpacemiT EP subgraphs. With
`ORT_ENABLE_ALL`, the dumped transformed EP graphs include all 102 Conv and
four MatMul operations from each source graph, while final TopK/output
housekeeping remains on CPU. FP16 also runs with `ORT_DISABLE_ALL`, unlike the
2.0.5 diagnostic.

The primary INT8 model does not create a runnable session. A dumped transformed
graph shows attempted partitioning, but compilation aborts before inference;
there is no executed INT8 provider placement or COCO surface.

## Minimal commands

The attachment contains generated tiny models, fixed inputs, independent
expected outputs, a neutral C++ runner, and an independent plugin smoke:

```bash
LD_LIBRARY_PATH="${ORT_ROOT}/lib" taskset -c 0 timeout 30s \
  ./ort_runtime_runner \
  --provider spacemit \
  --model models/B_qlinearconv.onnx \
  --input inputs/input_1x3x8x8_f32.bin \
  --output B.out \
  --opt-level disable \
  --intra-threads 1 --inter-threads 1

cmp B.out expected/B_independent_oracle.bin
```

Attachment: `ISSUE_1_RT206_MINIMAL_REPRO_BUNDLE.tar.gz`

SHA-256: `06387731f57e9703f0b7882b25d7cdaba141c924d28dd58522be6bb1dfa3e5f1`

## Open questions

1. Is a provider build newer than the 2.0.6 asset planned for explicit
   `kernel_shape`, `QLinearConv`, and `QLinearMatMul`?
2. Is the unchanged ORT core expected to dispatch the captured MLAS instruction
   safely on K1X/X60?
3. What numerical oracle should the packaged Track2 plugin sample satisfy?
4. Which provider log or API is authoritative for source-node assignment after
   transformed subgraph partitioning?
