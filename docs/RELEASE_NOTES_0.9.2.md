# Release Notes 0.9.2

0.9.2 is the final Stage59 maintenance release for the frozen YOLO26n-640
`K1X_INT8_V1` graph. It does not change executor arithmetic, quantization,
weights, graph structure, or model resolution.

## Fixed

- Restored the exact X60 release compiler contract in the top-level cross build;
  0.9.1 had accidentally omitted `-mtune=spacemit-x60 -funroll-loops` from the
  release objects.
- Added fail-closed official K1X configuration checks for IME, RVV target, and
  the frozen profile.
- Applied the per-handle busy contract to tensor metadata reads.
- Replaced unsafe demo signal handling with `sigaction` and bounded wakeups.
- Replaced ambiguous camera metric names with schema v2 and an explicit
  measured window.
- Added direct V4L2 MMAP sequence/timestamp telemetry.

## Camera Maintenance

- Public quality-wide and matched performance presets are distinct.
- `run_camera_demo_fast.sh` no longer silently selects the slower O2 camera
  diagnostic; it uses the measured 640x480 preset and a reversible, validated
  camera-only CPU5/xHCI IRQ profile.
- Recording defaults to a bounded asynchronous queue with one writer owner.
- Resize, letterbox, RGB, and annotated-frame buffers are reusable.
- Requested/backend-reported FPS, raw V4L2 buffer cadence, OpenCV decoded FPS,
  processed/displayed FPS, and application slot replacements are separate.

## Distribution

- Runtime archives contain the SDK and prepared model package but no source
  ONNX.
- Internal-R&D archives additionally contain the exact source ONNX and explicit
  provenance/license limitations.
- External source-ONNX redistribution remains not cleared.
- Project licensing remains `license-decision-pending` under the factual
  internal-use notice.

SONAME and C ABI remain `1`. This is an optimized engineering handoff, not
production certification and not a 20 FPS claim.
