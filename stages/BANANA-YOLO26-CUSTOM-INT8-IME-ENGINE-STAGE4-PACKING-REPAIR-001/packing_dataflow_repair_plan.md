# Packing Dataflow Repair Plan

## Stage 3 Problem

Stage 3 proved correctness, but Conv3x3 packing/im2col dominated measured time:

- Stage 3 Conv3x3 prepacked total: `149121 us`
- Stage 3 Conv3x3 packA/im2col probe: `157570 us`
- Stage 3 Conv3x3 scalar: `143708 us`

## Stage 4 Changes

1. Persistent weight prepacking:
   - `Y26PrepackedConvWeights`
   - aligned packed-B buffer
   - per-output-channel weight sums
   - source tensor name pointer and quant metadata pointer

2. Reusable workspace:
   - `Y26ConvWorkspace`
   - aligned A-tile buffer
   - explicit byte/peak byte introspection
   - created outside repeated Conv calls

3. A-panel repair:
   - A panels are packed as tile-contiguous `4x8` chunks.
   - The hot loop passes workspace tile pointers directly to `smt.vmadot`.
   - The previous extra copy from full `4xK` panel into local `std::array` per K tile was removed.
   - Conv1x1 packs directly from NHWC input.
   - Conv3x3 uses fused im2col-to-packA over small 4-output-position panels.

4. Loop-order evaluation:
   - `Y26_CONV_LOOP_ORDER_M_MAJOR`
   - `Y26_CONV_LOOP_ORDER_N_MAJOR`

## Decision

M-major is the selected Stage 4 path. It reuses one packed A panel across all output-channel tiles. N-major is correct but slower for the selected real shapes.
