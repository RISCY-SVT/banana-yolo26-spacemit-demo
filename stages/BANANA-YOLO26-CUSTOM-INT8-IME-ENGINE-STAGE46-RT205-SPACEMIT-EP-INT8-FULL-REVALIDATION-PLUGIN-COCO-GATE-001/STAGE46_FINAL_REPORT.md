# Stage46 final report

## Classification

`stage46-rt205-new-runtime-regression`

Start HEAD: `860ee58a55286f3e207d4b0eb7cec8a59a85bb9d`. End HEAD is recorded in the result packet and final
console response after the atomic local commit. Push: false.

## Proven

- The official RT205 archive is `ae512d21ef6a08a4db1a252237b7e2987978cb47ea4b1a86353ae54dc16ecae1` and passed safe extraction.
- The matched RT204 harness reproduces `output_type not implemented for clip minmax`.
- RT205 preserves that blocker and aborts after it on the primary first Conv.
- RT205 newly core dumps on QLinearConv and SIGILLs on QLinearMatMul controls that
  execute under RT204.
- A no-kernel-shape tiny Q/DQ Conv is assigned to RT205 EP and CPU/EP byte-exact;
  this bounded positive control does not generalize to the full model.
- RT204/RT205 CPU ORT_DISABLE_ALL are byte-exact to host ORT 1.27 for
  F0/F5/F6/F7 plus F8 blank and F9 structured edge fixtures.
- Stable primary CPU timing is `1023677.578412 us` on RT204 and
  `1024818.557872 us` on RT205; RT205 is `0.111459%` slower.
- The mandatory host full-COCO 2x2 matrix completed. FP32 disable/all mAP50-95 is
  `0.401438855549` / `0.401438842668`; INT8 is
  `0.372453424642` / `0.333615160723`.
- Full package CPU COCO rows: RT204 `0.374594101158669`, RT205 `0.374594101158669`.
- RT205 EP FP32/FP16 diagnostic timing is `445504.354768 / 368527.093944 us`;
  those discontinuous e2e outputs are not promoted as cross-runtime exact.
- RT205's SpacemiT plugin API and sample are package-present, but both the
  official and independent plugins have unresolved public ABI methods at load.
- All new board artifacts are under the NVMe stage root; eMMC exceptions: zero.

## Broken

- Historical explicit-kernel-shape Q/DQ Conv compilation under SpacemiT EP.
- RT205 QOperator Conv and MatMul execution (core dump/SIGILL).
- RT205 full primary INT8 SpacemiT-EP session/run.
- RT205 plugin loading and therefore plugin execution/partition/overhead gates.

## Unknown

- Full-model RT205 EP integer parity, COCO accuracy, and latency: not runnable.
- Plugin execution provider and partition preservation: loader failure prevents observation.
- Trained accuracy and measured board latency of the 416 and 512 student hypotheses.

## Correctness and accuracy policy

Host ORT 1.27 CPU with ORT_DISABLE_ALL plus independent operator semantics remains
the authority. Board package CPU rows validate integration. Board EP outputs must
also prove placement and semantics; CPU fallback is not accelerated INT8.

## Decision

Reject stock RT205 INT8 EP and its shipped plugin ABI for this model. Route the
next authorized work to K1X student 416/512 architecture and training preparation,
while retaining FP16/RVV as the fallback mainline. No training is authorized here.

## Validation

- Host Release build and CTest: pass, 44/44 tests.
- RT204 and RT205 matched-header RISC-V builds: pass.
- Independent and official plugin sample cross-builds: pass; loader gate fails as documented.
- Board loader identity, fixed fixtures, minimal repros, stable CPU timing, and
  full runnable COCO surfaces: executed.
- Python compile, storage-skill self-test, TSV structural validation, Git diff,
  staged diff, symlink, large-file, and secret/private-path checks: recorded in
  the final shared command ledger and result packet.

## Non-claims

No production readiness, default backend, model FPS, camera throughput, retained
custom-engine accuracy, usable RT205 plugin, student-model accuracy, or full
custom engine is claimed.
