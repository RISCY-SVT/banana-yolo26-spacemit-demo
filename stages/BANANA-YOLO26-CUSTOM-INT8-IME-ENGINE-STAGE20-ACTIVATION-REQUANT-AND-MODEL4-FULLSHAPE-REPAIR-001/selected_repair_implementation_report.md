# Selected Repair Implementation Report

selected_repair_lane: `C2`
selected_candidate: `C2_split0_concat_lut_4t`
implementation_file: `custom_int8_engine/tools/bench_stage20_model4_fullshape_c2f.cpp`

## Implementation

The C2 candidate keeps the same math and selected model4 C2f scope. It adds a boundary-specific SiLU LUT path from `/model.4/cv1/conv/Conv` output code to post-Concat Q/DQ signed int8 storage for the Split0 segment.

Instead of recomputing scalar `SiLU + QuantizeLinear` for Split0 inside the merge loop, the candidate generates `model4_cv1_concat_s8` with the existing RVV f32 LUT activation helper and copies the first channel span during Concat materialization.

## Scope

This is a Stage20 bench-sidecar implementation. It does not enable a default backend, full engine path, graph scheduler, OpenMP/all-core dispatch, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, or CPU4-7 IME.
