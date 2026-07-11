# Stage44 final report

## Identity

- stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE44-MODEL5-STRIDE2-PACK-AND-EXACT-REQUANT-REPAIR-001`
- classification: `stage44-model5-exact-no-net-win`
- repo: `/data/banana-yolo26-spacemit-demo`
- branch: `yolo26-custom-int8-engine`
- precheckpoint_head: `7a9b679f4b352c7894c9176539f1765d894daa73`
- stage43_checkpoint_commit: `f363c84ea39a8b6219ecec331b54e7deb892cf6a`
- stage44_end_head: `pending-local-commit-see-final-response-and-result-packet`
- pushed: `false`

## Proven

1. Stage43's dirty tree reconciled to 58 expected files and was preserved in the mandatory checkpoint commit.
2. Semantic policy remains independent integer semantics plus host ORT 1.27.0 CPU EP under `ORT_DISABLE_ALL`; board ORT is timing/integration only.
3. Resource-matched isolated model5 ORT is intra4: `11701.121842 +/- 31.218416 us`; intra1 continuity is `18169.770948 +/- 25.129083 us`.
4. The unchanged R0 custom worker matrix is exact; three workers are fastest among 1-4.
5. The previously missing model4-to-model5 R0 island is negative by `9301.556860 us` (`1.815641%`).
6. R2a exact stride-2 chunk fastpack is exact on F0-F7, workers1-4, and FRM RNE/RTZ/RDN/RUP/RMM.
7. In instrumentation-off ABBA, R2a improves R0 from `24636.0` to `24157.4 us`, a local `1.94291%` win.
8. R2a remains `106.4537%` slower than resource-matched ORT model5.
9. Final hybrid Path B remains slower: `515063.225590` versus `510864.440692 us`; delta `+4198.784898 us` (`+0.821898%`).
10. Workspace lifecycle now requires explicit init/magic/version; host CTest and ASan/UBSan pass.

## Broken

- Model5 custom compute is not competitive with equal-resource board ORT.
- Neither R0 nor R2a produces a positive paired contiguous-island/full-scaffold result.
- Diagnostic phase instrumentation perturbs total timing; those totals are not headline evidence.

## Unknown

- Whether a combined model5-6 island can remove enough additional boundary work to overcome the model5 kernel loss.
- Whether R3 output fusion or a different physical layout could win in a broader island; Stage44 evidence does not authorize those implementations.
- End-to-end custom-engine accuracy/performance; no full custom engine exists.

## Correctness and route

R2a is explicit and non-default. It preserves NHWC signed-code storage, existing s8xs8 `smt.vmadot`, explicit correction, per-channel fixed integer requant, and SiLU LUT. No CPU4-7 IME, new opcode, vmadot variant, or default dispatch was added. Disassembly and hashes are preserved in the stage packet.

## Timing status

All selection measurements use wall clock; process CPU time is separate. Warmup 10/runs 100/repeats 5 and CPU0-3 were used. Full scaffold paths alternate A/B, contain no reference execution, Python, file I/O, or internal model4-to-model5 transpose in the measured loop. No FPS claim is made.

## Decision

Stop model5 micro-tuning after Stage44. Keep R2a as experimental evidence only. Next stage should be `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-MODEL5-6-COMBINED-ISLAND-UPPER-BOUND-AND-ROADMAP-DECISION-001`, with no model6 implementation authorization.

Separately recommend a host-only accuracy stage using the fixed semantic contract, accepted preprocessing/postprocessing, and fixed model hash. No host mAP result may be transferred to a future custom board engine until end-to-end semantic equivalence is proven.

## Non-claims

No production readiness, model FPS, camera/full-frame performance, COCO/mAP, default dispatch, full custom engine, model6 implementation, push, or `/data/ncnn` mutation is claimed.
