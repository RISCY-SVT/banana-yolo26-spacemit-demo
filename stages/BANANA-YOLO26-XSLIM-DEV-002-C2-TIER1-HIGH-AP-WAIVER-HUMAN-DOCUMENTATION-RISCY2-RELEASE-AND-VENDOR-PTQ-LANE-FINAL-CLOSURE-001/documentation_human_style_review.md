# Documentation Human-Style Review

Status: `pass`.

The primary guides put purpose first, a minimal working command second, expected output third, common failure/fix next, and forensic detail later. They do not require Stage-ID knowledge.

Documented workflows cover generic PTQ, calibration preparation, configuration, quantization, structural validation, fixed fixtures, task evaluation and release validation. The K1X profile states signed per-tensor activation INT8 Q/DQ, symmetric per-channel Conv weights, no QLinear or UINT8 qparams, explicit Conv `kernel_shape`, six ordered outputs and an exact float tail. It does not turn parser acceptance into a placement claim.

The reconstruction guide says `BRECQ-inspired layer-local adaptive rounding infrastructure`, records the seven validated single-Conv targets, and marks full block/C2f/head reconstruction, QDrop, task-loss reconstruction and QAT unvalidated. YoloDecode source support is present, but B2/C2 do not use it; historical direct-E2E collapsed on 100/100, so the proven route remains six outputs plus the float tail.
