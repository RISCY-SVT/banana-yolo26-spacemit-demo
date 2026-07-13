# E2c compatibility

All 94 dense Conv rows satisfy the selected Q62 E2c invariants:

- right shift 62;
- positive multiplier;
- doubled M63 fits signed int64;
- int32 accumulator bound proven;
- output channels covered in exact four-channel groups plus exact tails;
- vCSR state saved and restored.

Dense E2c covers 2,665,292,800 MAC. Grouped/depthwise Conv and non-Conv
operators retain exact internal routes and are not mislabeled as E2c.
