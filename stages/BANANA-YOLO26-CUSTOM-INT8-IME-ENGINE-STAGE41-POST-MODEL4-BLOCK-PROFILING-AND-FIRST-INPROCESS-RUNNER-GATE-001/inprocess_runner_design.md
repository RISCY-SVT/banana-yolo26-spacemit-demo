# In-Process Runner Design

Stage41 adds `bench_stage41_inprocess_runner`, a C++ scaffold using the ONNX Runtime C API for fallback sessions and the existing custom `/model.4` C++ runner API for the custom island.

Measured runtime path:

```text
images tensor in memory
  -> ORT CPU prefix session
  -> NCHW uint8 to NHWC uint8 adapter
  -> y26_stage16_model4_c2f_run_cut_u8_output(...)
  -> NHWC uint8 to NCHW uint8 adapter
  -> ORT CPU suffix session
  -> output0 tensor in memory
```

Properties:

```text
python_in_runtime_path: false
per_block_file_io_in_runtime_path: false
custom_model4_called_in_memory: true
fallback_runtime: ONNX Runtime C API CPUExecutionProvider
default_backend_changed: false
production_runtime: false
model_fps_claim: false
```

Model/session files are loaded before measurement. Input/reference dump files are used only before or after the measured path for validation.

The board selected-mode hard gate is blocked by ORT CPU runtime contract mismatch, not by Python/file handoff.
