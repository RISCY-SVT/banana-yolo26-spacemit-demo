# Exact Q62 E2c contract

For every selected channel, the loaded package proves `right_shift=62`, a positive Q62
multiplier, and `(multiplier << 1) < 2^63`. E2c defines M63 as that shifted multiplier and uses
explicit `vsmul.vv` e64 under RNE. It preserves K1X_INT8_V1 exactly, performs no float or Q31
approximation, clamps and indexes the package LUT exactly, writes contiguous NCHWc8 C8 halves,
and saves/restores vector fixed-point CSR state.
