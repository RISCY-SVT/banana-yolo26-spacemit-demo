# QuantizeLinear Five-Case Oracle

An independent exact-rational round-to-nearest-even calculation was run for the five earliest Stage 42 divergent QuantizeLinear elements.

- Fixed host ORT matches the independent ONNX QuantizeLinear result in all five cases.
- Board vendor ORT is one uint8 code higher in all five cases.
- Distances from the exact quotient to the nearest half-integer range from `3.36e-7` to `4.37e-6`; these are near-half cases, not exact ties.
- Host `fenv` was RNE. The ambient board `fenv/frm` inside the closed vendor session was not observable.

This proves the expected result for these five inputs. It does not generalize provider assignment or identify the internal vendor-kernel implementation.
