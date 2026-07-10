# Native Runtime Architecture Plan

The final high-speed YOLO26 INT8 runtime should be native C++ only:

```text
python_in_final_inference_path: false
ORT_dependency_in_final_fast_path: false
per_block_files: false
default_backend_switch: false until correctness and coverage are proven
```

Architecture direction:

```text
1. Static graph skeleton with explicit block contracts.
2. Tensor arena allocator for persistent quantized tensors.
3. Per-tensor metadata: name, dtype, shape, layout, scale, zero point, storage zero point.
4. Prepacked weights stored once per custom block.
5. CPU0-3 cluster0 only for IME/vmadot work.
6. CPU4-7 may be considered only for future non-IME sidecars.
7. ORT CPU cuts remain debug/oracle/fallback scaffolding only.
8. Per-block oracle tests remain CI/debug artifacts.
```

No model FPS or production readiness can be claimed from Stage41.
