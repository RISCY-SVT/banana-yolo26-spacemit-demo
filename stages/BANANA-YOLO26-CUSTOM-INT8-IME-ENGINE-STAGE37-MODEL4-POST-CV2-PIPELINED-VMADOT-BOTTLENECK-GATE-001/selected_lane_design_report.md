# Selected Lane Design Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Mode

```text
enum: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
CLI: --merge-repair branch3x3_pipelined4
scope: selected /model.4 same-input ONNX-cut runner only
default backend changed: false
```

## Implementation Summary

Stage37 adds an explicit local mode that applies the Stage36 4-accumulator software-pipelined `smt.vmadot` MMT4D core to both selected branch 3x3 Conv nodes:

```text
/model.4/m.0/cv1/conv/Conv
/model.4/m.0/cv2/conv/Conv
```

The Stage36 pipelined path remains active for:

```text
/model.4/cv2/conv/Conv
```

## Files

```text
custom_int8_engine/include/y26_k1x_model4_c2f_runner.h
custom_int8_engine/include/y26_k1x_conv_kernels.h
custom_int8_engine/include/y26_k1x_threaded_conv.h
custom_int8_engine/kernels/conv_mmt4d_prepack.cpp
custom_int8_engine/kernels/conv_threaded.cpp
custom_int8_engine/src/model4_c2f_runner.cpp
custom_int8_engine/tools/bench_stage23_model4_runner_cut.cpp
```

## Kernel Policy

```text
instruction: smt.vmadot
storage: signed s8 x signed s8 -> s32
accumulator shape: Stage36 pipelined4 kernel shape
correction: existing explicit correction semantics preserved
threading: explicit cluster0 CPU0-3 only
mode selection: explicit local mode only
heap allocation in hot loop: no new hot-loop heap allocation
```

## Excluded Paths

```text
smt.vmadotus selected path: not used
vmadot1/2/3 direct/sliding integration: not used
vmadotn: not used
vfmadot / FP IME: not used
graph expansion: not done
full engine/default backend: not done
```
