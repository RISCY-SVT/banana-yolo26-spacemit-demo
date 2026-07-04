# Implementation Report

## Files Added

- `custom_int8_engine/include/y26_k1x_backbone_stage10_runner.h`
- `custom_int8_engine/src/backbone_stage10_runner.cpp`
- `custom_int8_engine/tests/stage10_backbone_expansion_fixture.h`
- `custom_int8_engine/tests/test_stage10_rvv_rounding_control.cpp`
- `custom_int8_engine/tests/test_stage10_backbone_expansion_runner.cpp`
- `custom_int8_engine/tools/extract_stage10_backbone_oracle.py`
- `custom_int8_engine/tools/bench_stage10_backbone_expansion.cpp`

## Files Modified

- `custom_int8_engine/kernels/activation_requant.cpp`
- `custom_int8_engine/CMakeLists.txt`
- `custom_int8_engine/tests/CMakeLists.txt`
- Stage 9 final report traceability-only end-head fix

## Runner Scope

The Stage 10 runner wraps the proven Stage 7/9 subset and adds only:

- Conv2 activation/requant LUT to Split output 1 scale domain
- Split channel slice `16..31`
- first branch Conv `/model.2/m.0/cv1/conv/Conv`

No graph-wide scheduler, full engine, ncnn mutation, XSlim use, or sliding vmadot implementation was added.
