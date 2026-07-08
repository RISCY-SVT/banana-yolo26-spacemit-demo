# Stage34 Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-VMADOT-CV2-PIPELINED-MICROKERNEL-001`

H1: Stage33 regression was caused by `smt.vmadotus` compute/copy overhead, not by the idea that IME is unusable.

H2: Current cv2 signed-storage `s8xs8 smt.vmadot` MMT4D compute is load-use/address-schedule limited on X60.

H3: Software pipelining with independent accumulator tiles and preloaded A/B tiles can reduce cv2 raw compute time without changing bytes.

H4: Keeping signed storage plus explicit correction avoids the Stage33 repack-copy penalty.

H5: If steady-state independent-vmadot throughput shows non-pipelined/blocking behavior, a large compute win is unlikely and the stage should stop with a ceiling report instead of forcing a bad kernel.

Non-claims:

```text
This is not full YOLO26 inference.
This is not model FPS.
This is not full-image/camera performance.
This is not COCO/mAP.
This is not production/default-backend readiness.
```
