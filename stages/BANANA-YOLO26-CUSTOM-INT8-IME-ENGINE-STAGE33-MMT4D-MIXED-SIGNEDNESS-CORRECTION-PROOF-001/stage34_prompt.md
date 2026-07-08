# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-THREAD-COPY-OR-OUTPUT-QUANTIZE-LOCAL-REPAIR-001

## Mission

Continue from Stage33. Do not expand the graph and do not implement a full YOLO26 engine.

Stage33 proved `smt.vmadotus u8xs8` correctness for `/model.4/cv2/conv/Conv`, but the candidate regressed selected-cut total time. The next stage should not select mixed signedness for this node.

Primary decision candidates:

```text
1. Reduce threaded Conv copy/thread overhead around `/model.4/cv2/conv/Conv`.
2. Revisit output QuantizeLinear bucket if it remains >15%.
3. If no local >5% selected-cut improvement is plausible, prepare selected-cut-to-next-block or full-model skeleton planning without performance claims.
```

Required gates:

```text
same-input ONNX-cut mismatches=0
FRM sweep pass
CPU0-3 only for IME
stable protocol warmup=10 runs=100 repeats=5
no full-model FPS / camera / COCO / production claim
```
