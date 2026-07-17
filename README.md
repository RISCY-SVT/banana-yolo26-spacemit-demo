# YOLO26 K1X INT8 Executor and Camera Demo

This repository contains the frozen `K1X_INT8_V1` executor for YOLO26n at a
fixed 640x640 letterboxed input, plus the `y26_k1x_demo` image/video/camera
application for the Banana-Pi BPI-F3.

The active runtime does not use ONNX Runtime or a vendor execution provider.
IME instructions execute only on workers pinned to CPU0-3; CPU4 is the
controller. The prepared runtime model is the versioned `package/` directory
shipped in the release archive.

## Quick Start

Build on the lab host:

```bash
scripts/build_cross.sh
```

Deploy the 0.9.2 release and run a camera:

```bash
scripts/deploy_to_banana.sh
scripts/run_camera_demo.sh
```

Run a headless image:

```bash
scripts/run_image_demo.sh /data/input.jpg /data/Screenshots/yolo26-output.png
```

See [CAMERA_DEMO_EN.md](docs/CAMERA_DEMO_EN.md),
[COLLEAGUE_FAQ_EN.md](docs/COLLEAGUE_FAQ_EN.md), and
[BUILDING_K1X_INT8_EXECUTOR.md](docs/BUILDING_K1X_INT8_EXECUTOR.md) for the
complete handoff. Russian instructions are provided alongside them.

## Scope

- Board: Banana-Pi BPI-F3 / SpacemiT K1X.
- Model tensor: RGB 640x640, exact letterbox with pad value 114.
- Output: 300 rows of `[x1, y1, x2, y2, confidence, class]`.
- Release status: optimized engineering handoff and camera demo ready.
- Not production certified. No 20 FPS claim is made.

The unchanged current graph is frozen. Model, quantization, training, student,
or co-design work requires a separate project and authorization.
