# Stage45 final report

- classification: `stage45-model-executor-codesign-recommended`
- stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-SYSTEM-ROOFLINE-VENDOR-ORT-FORENSICS-ACCURACY-AND-MODEL-CODESIGN-DECISION-001`
- start_head: `bdefd89cc4247cb9e0ddac6fd06b561b05d29c87`
- end_head: `pending-local-commit-see-final-response-and-result-packet`
- push: false
- production/default dispatch: not authorized and unchanged

## Proven

- Board `/data` is writable NVMe; all stage payloads stayed in `/data/k1x-stage-runs/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-SYSTEM-ROOFLINE-VENDOR-ORT-FORENSICS-ACCURACY-AND-MODEL-CODESIGN-DECISION-001`.
- Full vendor ORT CPU0-3 intra4: `461603.297250 +/- 435.600042 us`.
- Stage35 register vmadot throughput reproduced; realistic packed M12xN16 reached `54.360135 GMAC/s` with exact scalar-oracle output.
- Exact accepted graph: `2740153600` MAC, `5480307200` FLOPs-at-2/MAC, peak graph-order activation estimate `19660848` bytes.
- 500-image directional mAP50-95: FP32 `0.446714`, semantic INT8 `0.410683`, operational INT8 `0.373479`.

## Broken

- Current graph misses the 45 ms target even under the model5-geometry M12 compute-only ceiling (`50.407 ms`) before non-MAC work.
- Current semantic INT8 loses `3.603` AP points versus FP32 on the subset; operational optimization loses more.
- Stage44 custom model5 remains slower than resource-matched ORT and its contiguous scaffold was net negative.

## Unknown

- Vendor ORT model5 inner-kernel identity; vmadot code exists but execution was not localized.
- Full COCO val2017 accuracy, trained student accuracy, and actual student/AOT latency.
- Production camera/full-frame behavior.

## Decision

Select a K1X model/executor co-design stage: 416 student as latency-primary and 512
as accuracy fallback, distilled/QAT and paired with a resident-INT8 static AOT
executor. Do not expand the unchanged graph block by block. No model FPS,
production readiness, default dispatch, or retained-accuracy claim is made.

## Validation

- Python compileall: pass.
- Host Release build: pass; CTest 44/44 pass.
- RISC-V Release cross-build with the existing IME route: pass.
- Board CPU0 instruction controls and realistic microkernels: exact, no SIGILL.
- Board runtime loader: pass; deployed binary hash matches; no absolute RPATH.
- CPU0-3 policy: pass; CPU4-7 ran vendor-only ORT scouts, never custom IME.
- Git whitespace, symlink, large-file, and secret/private-path gates: pass after reviewed diagnostic-pattern self-matches.
