# Packing Dataflow Report

## Implemented

- Added MMT4D prepacked-B path for Conv1x1 and Conv3x3.
- Added per-output-channel weight-sum generation for zero-point correction.
- Added caller-owned 4-row A-panel workspace sized as `4 * align_up(K, 8)`.
- Added prepacked IME entrypoints:
  - `y26_conv1x1_i8s8s32_nhwc_ime_prepacked`
  - `y26_conv3x3_i8s8s32_nhwc_ime_prepacked`
- Prepacked IME loops do not allocate heap memory inside compute loops.
- Padding for quantized real graph uses signed storage zero point `za - 128`.

## Layout

- A panel: `4 x align_up(K,8)` signed `int8`, row-major, K-contiguous.
- B packed: `[n_tile][k_tile][n*8+k]`, where each tile is `4x8` signed `int8`.
- C tile: `4x4` signed `int32`.

## Current Performance Interpretation

The Stage 3 path removes repeated B packing and avoids repacking A once per output-channel tile.

- Conv1x1 benefits clearly from prepacked B and A-panel reuse.
- Conv3x3 is still dominated by im2col/A packing and border handling.
- `vmadot1/2/3` direct sliding ops were not used; direct-conv remains future work only.

## Caveat

The prepacked Conv3x3 benchmark uses quantized real-graph padding (`za`) while the old Stage 2 scalar/wrapper uses raw signed zero padding. Timings are comparable as loop-shape evidence; raw checksums differ at borders for that reason.
