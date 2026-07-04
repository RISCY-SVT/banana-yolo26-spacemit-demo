# SiLU LUT Exhaustive Oracle Report

classification: pass

Tests:

- `test_stage8_activation_requant`
- Act0 exhaustive `q in [0, 255]`
- Act1 exhaustive `q in [0, 255]`
- LUT output compared against the scalar reference formula using nearest-even quantization.

Result:

```text
host CTest: pass
board CPU0: pass
board CPU1: pass
board CPU2: pass
board CPU3: pass
mismatches: 0
```

The test also compares the `int8_lut` selected-subset runner handoff tensors and Conv2 output against the Stage 7 fixture oracle.
