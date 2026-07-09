# Pipelined Kernel Design Report

Target:

`/model.4/cv2/conv/Conv`, 1x1, `80x80x96 -> 80x80x128`.

Storage and math:

- activation storage: signed s8
- weight storage: signed s8
- instruction: base `smt.vmadot`
- accumulator: s32
- correction: existing explicit correction path
- rejected path not used: `smt.vmadotus`
- sliding variants not used: `smt.vmadot1/2/3`
- `vmadotn`: not used and not authorized

Implementation:

- New explicit local modes:
  - `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4`
  - `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED6`
- New single-thread sidecar entry:
  - `y26_conv2d_i8s8s32_nhwc_ime_prepacked_stage36_pipelined_v1`
- New threaded sidecar entry:
  - `y26_threaded_conv_run_ime_cluster0_stage36_pipelined_cv2`

Candidate A1:

- four independent accumulator groups
- accumulator pairs: `v20/v21`, `v22/v23`, `v24/v25`, `v26/v27`
- input vector: `v0`
- packed B vectors: `v1`, `v2`, `v3`, `v4`
- output channels per inner kernel group: 16

Candidate A2:

- six independent accumulator groups
- accumulator pairs: `v16/v17`, `v18/v19`, `v20/v21`, `v22/v23`, `v24/v25`, `v26/v27`
- input vector: `v0`
- packed B vectors: `v1` through `v6`
- output channels per inner kernel group: 24

Vtype policy:

- accumulator zero/store view: `e32,m2`
- steady `smt.vmadot` loop view: `e8,m1`
- vtype transitions are explicit around zeroing, steady compute, and store.

Restrictions preserved:

- no heap allocation in hot loop
- no CPU4-7 IME execution
- no graph expansion
- no global/default backend switch
