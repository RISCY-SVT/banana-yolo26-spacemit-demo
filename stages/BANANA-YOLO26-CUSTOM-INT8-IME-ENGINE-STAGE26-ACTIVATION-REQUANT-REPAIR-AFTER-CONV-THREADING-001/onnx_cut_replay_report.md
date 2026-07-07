# ONNX Cut Replay Report

The accepted candidate was run through the real `bench_stage23_model4_runner_cut` runner API against the same-input ONNX-cut expected output fixture.

```text
fixture_input_sha256: e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
mismatches: 0
max_abs_diff: 0
checksum: 106597930
affinity_ok: 1
```

This is same-input selected `/model.4` ONNX-cut evidence only. It is not full YOLO26 inference, model FPS, full-image/camera evidence, COCO/mAP, or production readiness.
