# Selected Repair Implementation Report

selected_lane: A3
selected_candidate: A3_branch1_add_lut

## Code Path

The repair adds explicit local merge mode:

```text
Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT
```

The mode is accepted only for the existing model4 C2f cut/full runner path. It is not a global backend, scheduler, graph expansion, or production/default dispatch.

## Mechanism

Prepare time:

```text
branch1_silu_f32_lut[256]
branch1_add_concat_lut_s8[256][256]
```

Hot loop:

```text
branch1 corrected int32 -> branch1 conv uint8 code using existing RNE quantize path
split1 uint8 code + branch1 conv uint8 code -> concat add-slot int8 via LUT
```

This removes per-element `std::exp` and per-element float Add/post-QDQ from the branch1 activation/add-slot hot path while preserving exact bytes against the same-input ONNX cut.
