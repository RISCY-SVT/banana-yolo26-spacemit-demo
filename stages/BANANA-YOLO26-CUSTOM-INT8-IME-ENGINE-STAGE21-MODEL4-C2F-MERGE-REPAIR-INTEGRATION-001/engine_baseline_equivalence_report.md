# Engine Baseline Equivalence Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Inspected Real Runner Files

```text
custom_int8_engine/include/y26_k1x_model4_c2f_runner.h
custom_int8_engine/src/model4_c2f_runner.cpp
custom_int8_engine/src/model4_branch_runner.cpp
custom_int8_engine/tools/bench_stage20_model4_fullshape_c2f.cpp
```

## Baseline Path Before Stage21

The real `/model.4` C2f runner already had `Y26Stage16MergeMode`, but `cfg->merge_mode` was only validated. The actual path in `run_after_stage15()` always used `build_concat_qdq_nhwc()`.

The pre-Stage21 real runner path for the first Concat segment did:

```text
/model.4/cv1 corrected int32
-> accumulator_to_silu_float()
-> quantize_concat_s8()
-> concat_s8 first segment
```

This matches the Stage20 `B1_threaded_branch0_4t` behavior, where split0 was recomputed for the Concat output scale inside merge handling.

## Stage20 C2 Transfer Expectation

Stage20 `C2_split0_concat_lut_4t` precomputed split0 directly at the Concat Q/DQ scale with a boundary-specific LUT:

```text
/model.4/cv1 corrected int32
-> y26_activation_requant_silu_int8_lut*_with_concat_scale
-> split0_concat_s8
-> concat_s8 first segment
```

The same transformation is local to the real runner because `model4_c2f_runner.cpp` already has access to:

```text
ws.stage15_ws.model4_cv1_i32
cfg.stage15.stage14.model4_cv1
cfg.concat_output_scale
cfg.concat_output_zero_point_u8
```

## Stage21 Integration

Stage21 added:

```text
Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT
ws.split0_concat_s8
ws.model4_cv1_to_concat_lut_s8
build_split0_concat_lut_activation()
```

The C2 mode is explicit and local to the model4 C2f runner. It does not switch the global backend or default dispatch.

## Equivalence Status

```text
source_inspection_equivalent_to_stage20_B1_baseline: yes
c2_transfer_expected: yes
host_compact_reference_vs_c2: pass
board_compact_reference_vs_c2: pass
```

The first implementation attempt incorrectly used `stage15_output_count`, which is branch0 output count, not `/model.4/cv1` output count. Host CTest caught this with nonzero mismatches. Stage21 fixed it to use `ws.stage15_ws.model4_cv1_output_count`; after the fix, host and board mismatches are zero.
