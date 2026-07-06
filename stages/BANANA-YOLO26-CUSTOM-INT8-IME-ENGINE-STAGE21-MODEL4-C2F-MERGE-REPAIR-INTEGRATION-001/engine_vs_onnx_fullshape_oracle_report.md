# Engine vs ONNX Full-Shape Oracle Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## ONNX Oracle Artifacts

Stage20 generated full-shape ONNX Runtime CPU tensors under:

```text
.deps/custom_int8_engine/stage20_fullshape_oracles/model4_c2f_synthetic_seeded/
```

Relevant ONNX boundary:

```text
name: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
path: .deps/custom_int8_engine/stage20_fullshape_oracles/model4_c2f_synthetic_seeded/model.4__cv2__conv__Conv_output_0_QuantizeLinear_Output.npy
dtype: uint8
shape: 1x128x80x80
sha256: 4e1221badad9df09be01cc8af03476ec9cc754ce1cfda9c5939aae93343ff33f
scale: 0.0660646632314
zero_point_u8: 142
```

## Stage21 Engine Comparison Status

```text
direct_engine_vs_full_model_onnx_boundary: partial
internal_engine_reference_exactness: pass
board_integrated_runner_mismatches: 0
stage20_fullshape_sidecar_mismatches: 0
```

Stage21 did not claim a direct engine-vs-full-model-ONNX equality pass for the integrated runner. The reason is structural:

```text
The Stage20/Stage21 representative full-shape timing fixture repeats compact internal model4 tensors.
The Stage20 ONNX full-shape tensors come from ONNX Runtime CPU with a full-model synthetic input.
Those two inputs are not the same tensor stream, so comparing their final outputs directly would be invalid.
```

The integrated real runner was proven bit-exact against the in-process scalar/reference path and compact oracle:

```text
board test_stage21_c2f_merge_repair:
  synthetic_seeded concat_mismatches: 0
  synthetic_seeded model4_cv2_mismatches: 0
  synthetic_gradient concat_mismatches: 0
  synthetic_gradient model4_cv2_mismatches: 0
```

The representative/full-shape Stage20-compatible C2 timing path remains exact against its scalar reference:

```text
candidate: C2_split0_concat_lut_4t
shape_class: representative_full_shape_model4_c2f_synthetic
mismatches: 0
checksum: -3094964234
```

## Required Follow-Up

A strict full-shape ONNX equality proof for the integrated runner needs a Stage22 oracle cut that feeds the exact same `/model.4/cv1` boundary tensor into both:

```text
1. a model4 C2f ONNX CPU subgraph or cut graph;
2. the integrated C++ model4 C2f runner.
```

Stage21 does not substitute internal self-consistency for that direct ONNX proof.
