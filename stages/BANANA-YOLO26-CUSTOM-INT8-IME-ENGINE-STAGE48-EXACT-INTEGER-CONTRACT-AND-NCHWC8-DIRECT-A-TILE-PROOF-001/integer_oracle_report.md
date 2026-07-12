
# K1X_INT8_V1 integer oracle report

`K1X_INT8_V1` is the exact authority. The package exporter derives integer
multipliers from the exact float32 bit patterns using rational arithmetic; the
board consumes only encoded integers and a 256-entry activation LUT. It does
not call `frexp`, `llround`, `exp`, or threshold search during prepare/run.

The Python reference uses a proven int64-safe full Conv accumulation and Python
arbitrary-precision requant arithmetic. Portable C++ scalar, board scalar, and
board IME are byte-exact for F0-F7. All 12 adversarial positive/negative tie,
threshold-neighborhood, saturation, and zero cases pass. The model5 absolute
accumulator bound is `36074272`, below INT32_MAX.

A complete second export to a distinct raw-evidence directory is byte-identical;
`diff -qr` reports no differences.

Legacy host ORT float-QDQ differs on F1, F2, F4, and one F7 element. That is a
separate model-replay diagnostic and is not the integer-contract authority.
