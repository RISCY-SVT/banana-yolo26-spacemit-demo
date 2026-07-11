# Stage 43 Final Report

classification: `stage43-model5-exact-no-compute-win`

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE43-CONTIGUOUS-MODEL5-8-ISLAND-ORACLE-AND-FIRST-BLOCK-GATE-001`

start_head: `7a9b679f4b352c7894c9176539f1765d894daa73`

end_head: `7a9b679f4b352c7894c9176539f1765d894daa73` (no commit)

## Proven

1. Graph-derived model5-8 semantic/quantized contracts, cuts, weights, and eight-fixture host oracle package are complete.
2. True isolated board profiles were measured for model5-8; cumulative-session subtraction was not used.
3. Model4 final activation, model5 host scalar, board scalar, and board IME are byte exact against the semantic fixed-host ORT 1.27 `ORT_DISABLE_ALL` cut for F0-F7.
4. Scalar and IME output hashes agree for all eight fixtures. FRM RNE/RTZ/RDN/RUP/RMM passes and restores ambient `frm`.
5. The model5 API consumes the existing model4 NHWC uint8 output directly. Internal model4-to-model5 transposes are zero.
6. Existing named `smt.vmadot` executes through the Stage37 four-accumulator route on CPU0-3; no SIGILL and `affinity_ok=1`.
7. The runtime tool no longer includes a test implementation source, and the RISC-V runner has no absolute build-tree RPATH.
8. Host build and all 44 CTests pass; RISC-V full cross build passes; focused ASan/UBSan test passes.

## Oracle Policy Finding

Policy B remains fixed-host based, with the Stage43 hierarchy made explicit:

- Level 0: independent Q/DQ/Conv integer semantics;
- semantic host cut: ORT 1.27.0 CPU EP with `ORT_DISABLE_ALL`;
- operational integration artifact: ORT 1.27.0 CPU EP with `ORT_ENABLE_ALL`;
- board ORT: integration/debug/timing only.

On stress fixtures, `ORT_ENABLE_ALL` differs from exact graph semantics: F1 2516 mismatches/max 76, F2 2595/max 64, F4 16342/max 71, F7 1/max 1. An independent adjacent u8xs8 pair/int16 saturation model reproduces sampled optimized-host Conv codes exactly. The K1X path does not emulate that x86-specific artifact. No tolerance is used.

## Timing

Same-session stable measurements, CPU0-3, performance governor, warmup 10, runs 100, repeats 5:

| Surface | Mean us | Status |
|---|---:|---|
| model4 final activation only | 2589.904 | measured |
| exact custom model5 Conv + postactivation | 26579.802 | measured |
| equivalent isolated board ORT model5 | 17862.861 | diagnostic runtime |
| custom delta | +8716.941 | 48.799% slower |
| island entry adapter | 3079.934 | measured separately |
| island exit adapter | 801.232 | measured separately |

Model5 internal compute is negative. The contiguous full-island and paired Stage42/Stage43 scaffold benchmarks were not run because the mandated model5 compute gate failed first. This avoids presenting adapter or suffix effects as a model5 compute win.

## Broken

- The bounded model5 path is not faster than equivalent board ORT.
- Host operational `ORT_ENABLE_ALL` is not a semantic oracle for adversarial full-range u8xs8 Conv inputs.
- The first RVV-float requant route had one F7 mismatch; the accepted exact diagnostic route uses prepared fixed requant and is slower.

## Unknown

- Whether a stride-2 fused pack and faster exact requant can remove the 8.717 ms model5 gap.
- Full model4-to-model5 hybrid scaffold ROI, intentionally short-circuited.
- Model6/model7/model8 custom performance; none were implemented.

## Authorization and Non-Claims

The exact implementation is experimental and not default-selected. The prompt does not authorize a commit for `stage43-model5-exact-no-compute-win`, so no commit or push was created.

This is not full custom YOLO26 inference, model FPS, camera/full-frame performance, COCO/mAP, production readiness, default dispatch readiness, or a full engine.

Next recommended stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE44-MODEL5-STRIDE2-PACK-AND-EXACT-REQUANT-REPAIR-001`.
