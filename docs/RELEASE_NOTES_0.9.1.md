# Release Notes 0.9.1

0.9.1 is release maintenance for the frozen YOLO26n-640 `K1X_INT8_V1`
executor. It does not change graph arithmetic, quantization, package identity,
or accepted predictions.

## Added

- `y26_k1x_demo` for image, video, and live V4L2 camera sources.
- Exact 640x640 letterbox/deletterbox and final top-300 visualization without a
  second NMS.
- Sequential and latest-frame camera policies, GUI/headless output, PNG capture,
  and MJPG AVI recording with a documented PNG fallback.
- ABI1 build-information query for release, source, contract, profile, package,
  and IME/RVV/frozen/RGB capability bits.
- Complete prepared-model SDK archives, camera evidence, bilingual FAQ, and
  object-size operating-envelope reports.

## Changed

- Project/package version is 0.9.1; SONAME remains 1.
- CMake package compatibility uses `SameMinorVersion` while major version is
  zero.
- Official K1X release builds fail capability validation if IME, RVV, or the
  frozen profile is missing.
- The active top-level build and public scripts use YOLO26 and the executor C
  ABI. The prior demo/runtime surface was removed; accepted stage evidence
  remains historical.

## Compatibility

All 13 Stage57 ABI1 functions remain unchanged. Two additive ABI1 functions
query build information, taking the public export count to 15. Existing ABI1
consumers and structure layouts are unchanged. The prepared package and
prediction identities remain exactly the Stage57 values.

This release is optimized-engineering-handoff-ready and camera-demo-ready. It
is not production certified and does not claim 20 FPS.
