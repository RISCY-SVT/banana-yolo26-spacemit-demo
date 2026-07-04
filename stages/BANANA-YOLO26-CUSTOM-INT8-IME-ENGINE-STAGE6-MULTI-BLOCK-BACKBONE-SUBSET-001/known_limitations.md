# Known Limitations

- Stage 6 is not a full YOLO26 engine and does not include a graph-wide scheduler.
- The selected subset stops at corrected int32 output of `/model.1/conv/Conv`.
- `/model.1` output Q/DQ, `/model.1/act/Sigmoid`, `/model.1/act/Mul`, `/model.2/cv1/conv/Conv`, and `/model.2/Split` are deferred.
- Activation/requant after Conv0 is a scalar float fallback and is the largest measured Stage 6 component.
- No COCO/mAP, camera path, full-image demo, model FPS, ncnn integration, XSlim, `vmadot1/2/3`, `vmadotn`, or FP/vfmadot work was done.
- Board execution remained cluster0-only for IME tests; CPU4-7 IME execution was not run.

