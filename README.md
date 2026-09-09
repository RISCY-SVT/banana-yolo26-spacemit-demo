# YOLO26 K1X INT8 Executor and Camera Demo

## XSlim Vendor Chapter Status

The vendor research chapter is frozen after documentation/metadata maintenance.
See the [source-publication addendum](stages/BANANA-YOLO26-XSLIM-MAINT-001-PUB-DUAL-REMOTE-BRANCH-PARITY-RESTORE-CHECK-AND-FINAL-CLOSURE-001/publication_status_addendum.md),
the historical [MAINT-001 closure](stages/BANANA-YOLO26-XSLIM-MAINT-001-D1-D3-REPOSITORY-DOCUMENTATION-FREEZE-AND-HANDOFF-001/CHAPTER_CLOSURE_RU.md)
and [operator summary](stages/BANANA-YOLO26-XSLIM-MAINT-001-D1-D3-REPOSITORY-DOCUMENTATION-FREEZE-AND-HANDOFF-001/MAINT001_OPERATOR_SUMMARY_EN.md).
XSlim `2.1.2+riscy.2.1` source-branch publication does not publish its local
candidate packages or create a maintenance release. B2 remains the
universal vendor control; C2 retains its existing separate TIER-1 higher-AP
waiver and historical universal FAIL. This does not change the custom executor,
runtime defaults, frozen models, or authorize a new campaign.

This repository contains the frozen `K1X_INT8_V1` executor and camera demo for
the Banana-Pi BPI-F3. R640 remains the only default and accepted release
profile. The integrated `0.10.0-internal-rd.1` research build also exposes eight
explicit opt-in Q0 profiles: R512, R448, R416, R384, R352, R320, R256, and R768.

The active runtime does not use ONNX Runtime or a vendor execution provider.
IME instructions execute only on workers pinned to CPU0-3; CPU4 is the
controller. The prepared runtime model is the versioned `package/` directory
shipped in the release archive.

## Quick Start

Build on the lab host:

```bash
scripts/build_cross.sh
```

Deploy the stable 0.9.3 R640 release and run a camera:

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
- Default model tensor: RGB 640x640, exact letterbox with pad value 114.
- Output: 300 rows of `[x1, y1, x2, y2, confidence, class]`.
- Release status: optimized engineering handoff and camera demo ready.
- Not production certified. No 20 FPS claim is made.

The unchanged current graph is frozen. Model, quantization, training, student,
or co-design work requires a separate project and authorization.

Non-R640 profiles are experimental Q0 evidence, are never auto-selected, and
are not deployment-promoted. See [PROFILE_GUIDE.md](docs/PROFILE_GUIDE.md) and
[LEGAL_STATUS.md](LEGAL_STATUS.md).
