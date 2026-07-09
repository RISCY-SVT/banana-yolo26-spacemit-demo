# Full-Model Skeleton Design

Stage40 implements a correctness-first Python-driven skeleton, not a production runtime API.

## Design Chosen

The practical route is:

```text
images
  -> ORT CPU prefix cut
  -> custom /model.4 C++ runner sidecar, explicit mode only
  -> ORT CPU suffix cut
  -> output0
```

The all-ORT fallback mode is:

```text
images
  -> ORT CPU prefix cut
  -> ORT CPU /model.4 cut
  -> ORT CPU suffix cut
  -> output0
```

This route was chosen because the repo already has proven model4 C++ runner APIs and host-side ONNX cut tooling. It avoids adding ONNX Runtime as a dependency of `y26_k1x_custom_int8_engine`.

## New Tools

| tool | purpose |
|---|---|
| `custom_int8_engine/tools/run_full_model_ort_reference.py` | full ORT CPU reference and boundary dump |
| `custom_int8_engine/tools/extract_block_cut_oracles.py` | prefix/model4/suffix cut extraction and skeleton comparison |
| `custom_int8_engine/tools/compare_tensor_outputs.py` | reusable tensor comparator |
| `custom_int8_engine/tools/extract_full_model_oracle.py` | compatibility wrapper for full-model oracle extraction |

## Explicit Non-Claims

This skeleton is file/boundary driven. Its timing is profiling evidence only and must be labelled `skeleton_total_latency_not_model_fps`.
