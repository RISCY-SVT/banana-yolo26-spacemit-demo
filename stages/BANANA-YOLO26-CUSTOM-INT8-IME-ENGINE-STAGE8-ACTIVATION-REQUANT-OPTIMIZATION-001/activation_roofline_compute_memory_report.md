# Activation Roofline Compute Memory Report

scope: selected-subset diagnostic only

## Tensor Sizes

| boundary | elements | approximate current read/write bytes |
|---|---:|---:|
| Act0 | 1,638,400 | int32 read 6.55 MiB, int8 write 1.56 MiB |
| Act1 | 819,200 | int32 read 3.12 MiB, int8 write 0.78 MiB |

## Observed Timings

| boundary | scalar-float fallback us | int8 LUT us |
|---|---:|---:|
| Act0 | 311321 | 128644 |
| Act1 | 154580 | 63924.9 |

## Diagnostic Conclusion

The LUT path is still far above a simple memory-copy floor. Remaining cost is dominated by per-element int32-to-conv-code quantization and scalar table lookup/write. This is not a whole-model roofline claim.
