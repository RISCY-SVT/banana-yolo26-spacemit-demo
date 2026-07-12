
# Stage48 final report

classification: stage48-integer-contract-pass-direct-layout-ort-competitive
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE48-EXACT-INTEGER-CONTRACT-AND-NCHWC8-DIRECT-A-TILE-PROOF-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 3c1eabb5198316c26c9577c0018343568e84c993
end_head: pending-local-commit-see-final-response

## Proven

- `K1X_INT8_V1_GENERAL` is versioned, package-defined, and exact across Python,
  portable C++ scalar, board scalar, and board IME for F0-F7 and adversarial ties.
- Model5 accumulator bound `36074272` proves int32 safety.
- `NCHWc8_SPATIAL_INNER_V1` byte order and C8 direct delivery pass; disassembly
  contains `vlseg2e64.v` and `smt.vmadot`; no generic per-byte interior pack exists.
- Selected M12/spatial/four-worker model5 is `6516.213018 us`, exact, and
  `4.049592x` faster than Stage47 R0; it is `44.310631%` below
  the resource-matched ORT reference.
- FRM RNE/RTZ/RDN/RUP/RMM produce one hash and restore RNE; CPU affinity is 0-3,
  CPU4-7 execute no IME, and no SIGILL occurred.

## Broken

- Scalar NCHW/NCHWc8 entry/exit conversions remain expensive and are not an
  acceptable per-operator path.
- Legacy float-QDQ replay is not byte-exact to the new integer contract on all fixtures.

## Unknown

- A persistent NCHWc8 contiguous slice has not yet been measured.
- An exact RVV epilogue was not implemented because the scalar exact epilogue
  already crossed the predeclared ORT-competitive threshold.
- Full-model integer-contract accuracy and performance remain unknown.

## Decision

The specific direct-layout hypothesis passes. Keep the executor-first lane open
for one persistent NCHWc8 contiguous-slice/LUT-v2 gate. Student 416 and 512 both
remain deferred; training is unauthorized. RT205 work performed: false.

## Validation

- Host configure/build and all 47 CTests: pass.
- Focused x86 ASan/UBSan Stage48 tests: 2/2 pass.
- Python compile and deterministic package regeneration: pass; directory diff is empty.
- Full RISC-V cross-build and board loader: pass; no RPATH/RUNPATH.
- Board scalar/IME F0-F7, adversarial ties, M12 tail, FRM restoration,
  CPU0-3 affinity, byte order, and 10/100/5 performance matrix: pass.
- Git diff checks, symlink, large-file, secret, private-path, and `/data/ncnn`
  integrity checks: pass.

## Non-claims

No production readiness, full engine, default dispatch, model FPS, camera
performance, COCO accuracy, or trained-student accuracy is claimed.
