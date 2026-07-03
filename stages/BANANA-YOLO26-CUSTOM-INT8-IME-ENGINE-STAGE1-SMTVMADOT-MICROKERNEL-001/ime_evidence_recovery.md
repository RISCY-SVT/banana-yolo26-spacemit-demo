# IME Evidence Recovery

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001

## Accepted local evidence read

- Stage 0 repo reports:
  - `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001/final-report.md`
  - `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001/summary.ru.md`
  - `static_model_format_v0.md`, `quantization_and_scales_plan.md`, `signedness_zero_point_audit.tsv`
  - `first_layer_shape_tile_plan.md`, `cluster0_safety_plan.md`, `oracle_plan.md`, `performance_gate_plan.md`
- Accepted control evidence:
  - `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/2026-06-27_07-00-01_W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001-TIER2-REVIEW-AND-CLOSURE-001/accepted-facts.md`
  - `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/2026-06-27_07-00-01_W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001-TIER2-REVIEW-AND-CLOSURE-001/final-report.md`
- Accepted exchange/task-run evidence:
  - `/exchange/results/archive/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001`
  - `/data/lab/task-runs/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001`
  - `/data/ncnn-logs/ai-team/2026-06-26/2026-06-26_09-59-28__contcodex__W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001__int8-vmadot-toolchain-hardware-proof`
  - `/data/ncnn-logs/ai-team/2026-06-26/2026-06-26_11-09-43__contcodex__W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001__int8-vmadot-toolchain-hardware-proof`

## Recovered facts used by Stage 1

- `smt.vmadot` is the only Stage 1 implementation primitive.
- Operation: signed int8 x signed int8 to signed int32 accumulation.
- Tile contract: 4x4x8 on X60/VLEN=256/int8.
- Accepted execution surface: cluster0 CPUs 0-3 only.
- CPU4/CPU5 accepted evidence: controlled SIGILL for tested path.
- Exact scalar oracle evidence exists for accepted `smt.vmadot` cases.
- `vmadotn` is rejected/not authorized on tested routes.
- `smt.vmadot3` has board execution evidence, but not an independent oracle acceptance claim for Stage 1 implementation.
- FP/vfmadot remains blocked/deferred.

No conflict was found between current prompt, Stage 0 outputs, and accepted 0003 evidence.

## xslim note

`xslim` was not used as a model source, authority source, build path, or validation path in this stage.
