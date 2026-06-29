# banana-yolo26-spacemit-demo

Isolated R&D workspace for Banana Pi BPI-F3 / SpacemiT K1X YOLO26
experiments using SpacemiT ONNX Runtime.

This repository was bootstrapped from the production
`banana-yolo11-spacemit-demo` release tag and is intentionally separated from
the frozen YOLO11 production repository. Use this tree only for YOLO26,
SpacemiT ORT 2.0.4, K1/K3 runtime architecture-selection, API/provider-option,
decode-contract, and model/runtime compatibility work.

See `TEMPLATE_SOURCE.md` for the exact source tag/commit used to seed this
workspace.

## R&D status

- Production claims: none.
- Production remotes: disabled; the inherited remote is renamed to
  `template-yolo11-gitlab`.
- Current focus: YOLO26n export/decode contract and SpacemiT ORT 2.0.4 runtime
  forensics.
- Frozen production baseline: the YOLO11 project remains
  `banana-yolo11-spacemit-demo` at tag `production-2026-07-02`.

## Frozen YOLO11 production policy, for comparison only

- Primary image visual: generated `dynamic640` INT8 on `rt201`.
- Normal camera: generated `dynamic640` INT8 on `rt201`.
- Fast-live camera: vendor320 INT8 on `rt123`, `320x320` letterbox.
- Vendor320 trusted visual: `rt123`.
- Vendor320 low-latency perf: raw `rt201`, perf-only.
- Vendor320 `rt201` visual workaround: available, SHA256-guarded, non-default.
- FP16: experimental only, `keep_io` 640 on `rt201`/`rt202b1`.
- YOLO26n: P2 only in the production repo, not production.
- Stable `rt202`: staged/tested, not adopted in the production repo.

## Template notes

The remainder of this README still contains template-era YOLO11 operational
details. Treat them as inherited helper documentation until the R&D code path is
renamed and narrowed around YOLO26.

---

## Inherited YOLO11 Template Reference

Standalone C++ demo repository for Banana Pi BPI-F3 / SpacemiT K1X using the vendor-tested ONNX Runtime stack:

- validated vendor runtimes:
  - `spacemit-ort.riscv64.1.2.3` for trustworthy vendor320 visual inference
  - `spacemit-ort.riscv64.2.0.1` for vendor320 low-latency benchmarking and dynamic640
  - `spacemit-ort.riscv64.2.0.2` is fetchable for regression, but Day 2
    validation did not adopt it because it aborts on the current model paths
- closed execution providers:
  - `libspacemit_ep.so.1.2.3`
  - `libspacemit_ep.so.2.0.1`
  - `libspacemit_ep.so.2.0.2` for explicit non-default stable-runtime regression
- model family: Ultralytics YOLO11n
- primary optimized path: INT8 ONNX on SpaceMIT EP

The repository is designed to be usable by another engineer from scratch once the canonical K1X toolchain and sysroot overlay are already installed.

## Production policy for 2026-07-02

- Primary production visual branch: generated `dynamic640` INT8 on `rt201`.
- Default image visual path: generated `dynamic640` INT8 on `rt201`.
- Default normal camera path: generated `dynamic640` INT8 on `rt201`.
- Fast-live camera path: official vendor320 INT8 on `rt123`, `320x320` letterbox, with the camera requested at `640x480`.
- Vendor320 trusted visual path: official vendor320 INT8 on `rt123`.
- Vendor320 low-latency benchmark path: raw `rt201`; this is perf-only, not the default visual path.
- Vendor320 `rt201` visual workaround: available and SHA256-guarded, but slower than the `rt123` visual path.
- FP16: experimental coverage only; the usable board-side path is `keep_io` FP16 `640x640` on `rt201`/`rt202b1`.
  Stable `rt202` was evaluated on Day 2 and is not a replacement for `rt202b1`.

The Day 2 release-candidate regression confirmed this policy after the Day 0
loader recovery fix and Day 1 full regression.

For the consolidated management-facing FPS/latency table, including metric-class
caveats and rejected/P2 paths, see `docs/FPS_SUMMARY.md`.

## Architecture

- Host build: Ubuntu 24.04 x86_64 cross-build container
- Target board: Banana Pi BPI-F3 / SpacemiT K1X
- Runtime matrix:
  - vendor320 visual path: `rt123` = `spacemit-ort.riscv64.1.2.3`
  - vendor320 perf path: `rt201` = `spacemit-ort.riscv64.2.0.1`
  - primary production dynamic640 path: `rt201` = `spacemit-ort.riscv64.2.0.1`
  - fast-live camera path: trusted vendor320 visual stack on `rt123`
  - stable-runtime evaluation path: `rt202` = `spacemit-ort.riscv64.2.0.2`
    (explicit test tag only, not a production default)
- Benchmark model path:
  - official vendor YOLO11n INT8 320x320 ONNX
- Primary/default visual demo path:
  - Ultralytics ONNX export
  - xquant-based INT8 conversion for custom sizes such as 640x640
  - generated model: `models/generated/xquant_640/yolov11n_640x640.dynamic_int8.onnx`

The application supports:

- single-image inference
- USB camera live inference
- explicit fast-live camera mode
- headless and display modes
- forward-only benchmarking
- full pipeline benchmarking
- annotated image/video output
- per-stage metrics logging
- explicit camera capture/decode timing separate from inference timing

The repository is intentionally usable in two modes:

- host-wrapper mode from the x86_64 cross-build container
- board-local mode directly on Banana after deploy

## Project layout

- `src/`, `include/`: C++ application
- `cmake/`: cross toolchain and vendor runtime discovery
- `scripts/`: fetch/build/deploy/run helpers
- `configs/xquant/`: xquant templates
- `third_party_manifest/`: pinned runtime/model metadata
- `docs/`: results and troubleshooting
- `scripts/reference/build_scripts/`: imported local helper scripts

## External references used

- Vendor runtime docs: `https://bianbu.spacemit.com/en/ai/onnxruntime`
- Vendor C++ example docs: `https://bianbu.spacemit.com/en/brdk/Model_deployment/4.3_CPP_Inference_Example`
- Demo overview: `https://bianbu.spacemit.com/en/ai/spacemit-demo`
- Vendor runtime archive: `https://archive.spacemit.com/spacemit-ai/onnxruntime/`
- Ultralytics export docs: `https://docs.ultralytics.com/modes/export/`
- xquant docs: `https://bianbu.spacemit.com/en/brdk/Advanced_development/7.1_Model_Quantization`
- SpacemiT deployment pipeline docs: `https://bianbu.spacemit.com/en/brdk/Model_deployment/4.4_Training_and_Deployment_Pipeline`

## Toolchain prerequisites

This project expects the canonical local K1X environment:

- toolchain root: `/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2`
- base sysroot: `${TOOLCHAIN_ROOT}/sysroot`
- overlay sysroot: `/data/sysroots/k1x-gtk3-overlay`

Source the environment first:

```bash
source /data/build_scripts/01-env.sh
```

## Sysroot overlay and local K1X helper scripts

Import the local helper scripts into this repository:

```bash
./scripts/import_local_k1x_scripts.sh
```

Refresh the overlay sysroot from the board:

```bash
./scripts/prepare_overlay_from_local.sh
```

## Vendor runtime

Fetch and stage the vendor runtime:

```bash
./scripts/fetch_vendor_runtime.sh
```

The validated runtime matrix is pinned in `third_party_manifest/runtime.lock`.
The fetch helper stages the public tarballs required for production plus
historical/experimental regression:

- `rt123` -> `spacemit-ort.riscv64.1.2.3`
- `rt201` -> `spacemit-ort.riscv64.2.0.1`
- `rt202b1` -> `spacemit-ort.riscv64.2.0.2+beta1`
- `rt202` -> `spacemit-ort.riscv64.2.0.2`

Day 2 evaluated final stable `rt202` as a bounded release-candidate check. It
is kept fetchable and selectable for reproducibility, but it is not an adopted
runtime path because it aborted on the current dynamic640, FP16 640, and
vendor320 test cases even after a board reboot with an idle `/dev/tcm`.

Day 3 adds an operator handoff package under `release/`. Start with
`release/DEMO_COMMANDS.md` for supported commands and
`release/PRODUCTION_READINESS_REPORT.md` for the release-candidate scope.

## Models

Fetch vendor-provided YOLO11n 320x320 models:

```bash
./scripts/fetch_models.sh
```

Outputs land under `models/vendor/yolo11/`.

Notes:

- The official vendor INT8 320x320 model is restored as a trustworthy visual path in this repository, but only on the validated `rt123` stack (`spacemit-ort.riscv64.1.2.3`) with letterbox preprocessing.
- The same vendor320 model remains the low-latency benchmark path on `rt201` (`spacemit-ort.riscv64.2.0.1`).
- A focused EP/runtime pass on 2026-03-15 found a public `rt201` visual workaround for vendor320:
  - `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE=1`
  - `SPACEMIT_EP_DISABLE_OP_NAME_FILTER=/model.23/Slice;/model.23/Slice_1;/model.23/Add_1;/model.23/Add_2;/model.23/Sub;/model.23/Sub_1`
  - this restores semantically good vendor320 detections on the canonical photo and on blank-white sanity input
  - it is much slower than `rt123`, so it is not the default visual policy
- The default visual demo path remains the generated 640x640 dynamic INT8 model, because it is the highest-quality user-facing path on the public stack.
- A focused 2026-03-14 root-cause pass showed:
  - the public vendor YOLO11 C++ example is semantically good on `1.2.2` and `1.2.3`
  - the same public example becomes semantically poor on `2.0.1`
  - our repository now mirrors that runtime split explicitly instead of pretending a single tarball version works for every model/path
- A follow-up clean-room runtime-line pass confirmed:
  - `rt201` (`2.0.1`) remains semantically bad for vendor320 even when `/dev/tcm` is clean and no `alloc failed(...)` appears
  - `rt202b1` (`2.0.2+beta1`) behaves like `rt201` for vendor320 and does not restore correct detections
  - public `1.2.4` package line is semantically good for vendor320, but still breaks the dynamic640 path
  - a chain-complete public compatibility pass then tested the official public example path with:
    - exact same `yolov11n_320x320.q.onnx` bundle
    - exact same canonical photo input hash
    - decode contracts `centerwh` and `xyxy`
    - graph optimization levels `0`, `1`, `2`, and `99`
  - result of that pass:
    - no public `2.0.x` combination restored a good vendor320 image result
    - `rt201`, tarball `rt202b1`, and package `pkg202fix` all remained bad on the same public `q.onnx` bundle
    - the public float `yolov11n_320x320.onnx` bundle also failed on `2.0.x` with EP-side reshape/compile errors instead of becoming a viable fallback
  - therefore the repository keeps the current policy:
    - vendor320 visual -> `rt123`
    - vendor320 perf -> `rt201`
    - dynamic640 -> `rt201`
  - a later EP/runtime-localization pass refined that conclusion:
    - `rt201` can be made visually correct only with the explicit public workaround above
    - the workaround is far slower than `rt123`, so `rt123` remains the default vendor320 visual runtime
    - `rt202b1` still remains bad even with the same workaround
- No official 640x640 vendor INT8 URL is currently pinned, so 640 uses the custom export + xquant path.
- In practice, the fast and reproducible 640 path in this repository is the `xquant` dynamic INT8 fallback. Public static calibration was attempted but remained too slow for a practical demo workflow.

## FP16 model coverage

The repository also carries a reproducible FP16 investigation path for YOLO11n `320x320` and `640x640`.

Generate the current repo-managed FP16 models:

```bash
./scripts/fetch_or_build_fp16_models.sh
```

Model families:

- `models/generated/fp16/yolov11n_<size>x<size>.fp16.onnx`
  - true FP16 I/O
  - dtype-validated
  - currently not recommended on the tested public board runtimes because they crashed or corrupted during execution
- `models/generated/fp16/yolov11n_<size>x<size>.fp16_iop32.onnx`
  - internal FP16 weights with FP32 I/O
  - dtype-validated
  - this is the only practical board-side FP16 test chain currently exercised by the repo

Run the reproducible FP16 matrix helper:

```bash
./scripts/bench_fp16_matrix.sh
```

Notes:

- the helper defaults to `FP16_MODEL_VARIANT=keep_io`
- use `FP16_MODEL_VARIANT=full` only for explicit runtime research, not as a recommended board path

Current honest result summary:

- public official/vendor YOLO11n FP16 ONNX artifacts were not found for `320` or `640`
- Ultralytics `half=True` export on CPU was tested and rejected because it still produced FP32 ONNX
- `keep_io` FP16 `640x640` works on `rt201` and `rt202b1`
- stable `rt202` was evaluated and did not replace `rt202b1`; it aborted on
  the same FP16 640 path that `rt202b1` runs successfully
- `keep_io` FP16 `320x320` fails on all three validated public runtime lines
- `keep_io` FP16 also fails on `rt123` for `640x640`
- full-I/O FP16 models are dtype-correct, but they are not a supported board path on the tested public vendor lines

See `docs/RESULTS.md` for the measured FP16 tables and exact failure notes.

## Optional custom export and quantization

Export YOLO11n from Ultralytics:

```bash
./scripts/export_ultralytics_onnx.sh 640 yolo11n.pt
```

Quantize with xquant static calibration:

```bash
CALIB_COUNT=10 ./scripts/quantize_xquant.sh 640 /path/to/yolo11n_640x640.onnx yolov11n_640x640.q
```

Fast fallback dynamic quantization:

```bash
XQUANT_MODE=dynamic ./scripts/quantize_xquant.sh 640 /path/to/yolo11n_640x640.onnx yolov11n_640x640.dynamic_int8
```

If an existing working xquant environment is available, reuse it explicitly:

```bash
XQUANT_PYTHON=/data/ort-spacemit-track/quant/venv/bin/python3 \
CALIB_COUNT=10 \
./scripts/quantize_xquant.sh 640 /path/to/yolo11n_640x640.onnx yolov11n_640x640.q
```

## Cross-build

```bash
./scripts/build_cross.sh
```

The build helper now checks OpenCV explicitly through `./scripts/ensure_opencv.sh`.
This repository does not vendor OpenCV. It expects the canonical local cross install
at `/data/opencv/install-k1x-gtk3`, and deploy stages the matching runtime libraries
under the board-side repo root.

## Deploy to Banana

```bash
./scripts/deploy_to_banana.sh
```

## Run image demo

```bash
./scripts/run_image_demo.sh
```

The no-argument image demo uses the default visual path:

- model: `models/generated/xquant_640/yolov11n_640x640.dynamic_int8.onnx`
- input size: `640`
- confidence: `0.25`
- runtime tag: `rt201`

If you explicitly override the model to `models/vendor/yolo11/yolov11n_320x320.q.onnx`, the script auto-selects runtime `rt123` and restores the validated vendor320 visual path. If you explicitly force `BANANA_DEMO_RUNTIME_TAG=rt201`, the script now auto-enables the validated public workaround for visual correctness. Disable that workaround only when you intentionally want the raw low-latency perf path:

```bash
BANANA_DEMO_RUNTIME_TAG=rt201 BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX=0 ./scripts/run_image_demo.sh /path/to/image.jpg models/vendor/yolo11/yolov11n_320x320.q.onnx 320
```

The image helper accepts optional positional overrides:

```bash
./scripts/run_image_demo.sh <image> <model> <input_size> <conf>
```

Board-local image runs now default to `DISPLAY_FLAG=auto`:

- if a GUI session is already exported, the helper keeps that display path
- if the board shell is a plain tty but Wayland/X11 sockets are present, the helper attempts to restore the local GUI env automatically
- if neither is available, the helper stays headless and the application logs that fallback explicitly

Board-local direct execution after deploy:

```bash
cd /home/svt/banana-yolo11-spacemit-demo
BANANA_DEMO_EXEC_MODE=board ./scripts/run_image_demo.sh
```

## Run camera demo

```bash
./scripts/detect_camera_formats.sh
./scripts/run_camera_demo.sh
```

Board-local camera default behavior:

- `DISPLAY_FLAG=auto`
- `HEADLESS_FLAG=auto`
- `MAX_FRAMES=0`
- live preview stays open until `q` / `ESC` or `Ctrl-C`
- the app logs the resolved HighGUI backend, capture backend, and camera open method
- the app also logs the resolved affinity policy, the main thread TID, and the cluster0/cluster1 CPU lists used for the run
- the first raw frame is pushed to the preview immediately with a warmup banner while the first inference initializes
- if GUI env/backend is not usable, the helper prints an explicit headless fallback message and the app emits periodic progress logs instead of appearing stuck
- saving a still image is supported by pointing `SAVE_OUTPUT` at `.jpg`, `.jpeg`, `.png`, or `.bmp`

Host-wrapper camera default behavior:

- `DISPLAY_FLAG=0`
- `HEADLESS_FLAG=auto`
- `MAX_FRAMES=200`

## Run fast-live camera demo

```bash
./scripts/run_camera_demo_fast.sh
```

Fast-live camera behavior:

- model: official vendor320 INT8 ONNX
- runtime: `rt123`
- preprocess: letterbox
- input size: `320`
- default camera request: `640x480 @ 60`
- purpose: better live responsiveness than the default dynamic640 path
- trade-off: lower spatial detail than the default 640 visual path

Measured on the current USB camera:

- default dynamic640 live path, steady post-warmup:
  - `capture_ms ~= 48-50`
  - `inference_ms ~= 206-210`
  - `total_ms ~= 254-260`
  - practical steady preview rate: about `3.8-3.9 FPS`
- fast-live vendor320 path, steady post-warmup:
  - `capture_ms ~= 1.1-1.2`
  - `inference_ms ~= 50-53`
  - `total_ms ~= 67-69`
  - practical steady preview rate: about `14.4-14.9 FPS`

The current fast-live defaults were chosen from measured candidates:

- `640x480` request is the preferred setting on the current camera
- `1280x720` request forced the camera into `YUYV @ 7.5 FPS` and dropped the effective loop rate to roughly `1 FPS`
- therefore the repo intentionally keeps `640x480` as the fast-live default instead of pretending the larger capture size is still responsive

Useful environment overrides:

```bash
DISPLAY_FLAG=1 CAMERA_PIXFMT=mjpg CONFIDENCE=0.25 ./scripts/run_camera_demo.sh /dev/video20
```

Fast-live override example:

```bash
DISPLAY_FLAG=1 ./scripts/run_camera_demo_fast.sh /dev/video20
```

Runtime override:

```bash
BANANA_DEMO_RUNTIME_TAG=rt123 ./scripts/run_camera_demo.sh auto /home/svt/banana-yolo11-spacemit-demo/models/vendor/yolo11/yolov11n_320x320.q.onnx 320
```

By default the camera helper does not record video. Recording is opt-in:

```bash
SAVE_OUTPUT_REMOTE=/home/svt/banana-yolo11-spacemit-demo/outputs/camera_320.avi ./scripts/run_camera_demo.sh /dev/video20
```

Still-image capture from camera mode is also supported:

```bash
DISPLAY_FLAG=0 HEADLESS_FLAG=1 MAX_FRAMES=1 SAVE_OUTPUT=/home/svt/banana-yolo11-spacemit-demo/outputs/camera_fast.jpg ./scripts/run_camera_demo_fast.sh
```

Backend sanity probe from Banana:

```bash
python3 ./scripts/probe_opencv_ui.py
```

Compact affinity trace capture:

```bash
./scripts/capture_camera_affinity.sh
./scripts/capture_camera_affinity.sh fast
```

- The helper saves run logs, `mpstat`, `pidstat`, `ps -L`, and per-thread `Cpus_allowed_list` snapshots into one bundle.
- Use it when you want a reproducible board-side view of camera thread placement without rebuilding the application.

Board-local direct execution after deploy:

```bash
cd /home/svt/banana-yolo11-spacemit-demo
BANANA_DEMO_EXEC_MODE=board ./scripts/run_camera_demo.sh
```

## Generate API docs

```bash
./scripts/gen_doxygen.sh
```

If `doxygen` is not installed system-wide, the helper downloads a user-local
Ubuntu package set into `.cache/doxygen-ubuntu/` and runs it from there.

Verify literal file-level coverage with:

```bash
./scripts/check_doxygen_coverage.sh
```

If you explicitly override the model to the vendor 320x320 INT8 ONNX, the script auto-selects `rt123`. If you explicitly force `BANANA_DEMO_RUNTIME_TAG=rt201`, the visual helpers now auto-enable the validated public workaround. Disable it with `BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX=0` only when you intentionally want the raw low-latency perf stack.

## Benchmark

Forward-only:

```bash
./scripts/bench_forward_only.sh
```

Full pipeline:

```bash
./scripts/bench_full_demo.sh
```

The forward-only benchmark compares the application against vendor `onnxruntime_perf_test`, because vendor CV tables exclude preprocess and postprocess.

Benchmark runtime policy:

- `bench_forward_only.sh` defaults to the low-latency `rt201` stack for vendor320 benchmarking
- `bench_full_demo.sh` defaults to the validated visual stack (`rt123` for vendor320, `rt201` otherwise)
- override either script with `BANANA_DEMO_RUNTIME_TAG=rt123|rt201` when you need a specific matrix entry
- only the visual helpers auto-enable the slower vendor320 `rt201` workaround; forward-only benchmarking keeps the raw perf stack unless you export the workaround variables yourself

Affinity policy notes:

- The default product path still uses a single `cluster0` mask for the demo process.
- This keeps ORT worker-thread inheritance deterministic and preserves the measured camera/live behavior already validated in this repo.
- Two stricter split strategies were tested on the board:
  - per-frame migration of the main thread between helper work on `cluster1` and inference on `cluster0`
  - a dedicated synchronous inference worker on `cluster0` with UI/camera work kept on `cluster1`
- Both variants were rejected for the product default because they did not improve measured live FPS on the current camera/display stack, and one variant introduced long-tail inference stalls.
- Use `./scripts/capture_camera_affinity.sh` if you need to reproduce those measurements or check a different board image.

## CLI highlights

The binary supports:

- `--model`
- `--labels`
- `--input-size 320|640`
- `--source image:<path>|camera:auto|camera:/dev/videoN|camera:<index>`
- `--provider spacemit|cpu`
- `--pin cluster0|cluster1|none|list:<csv>`
- `--threads`
- `--conf`, `--iou`
- `--display`, `--headless`
- `--save-output`
- `--log-file`
- `--benchmark-only`
- `--benchmark-mode forward|full`
- `--camera-width`, `--camera-height`, `--camera-fps`, `--camera-pixfmt`
- `--decode-mode auto|vendor|ultralytics`
- `--warmup`, `--runs`, `--repeats`

## Known-good defaults

- Default visual demo model:
  - `models/generated/xquant_640/yolov11n_640x640.dynamic_int8.onnx`
- Default fast-live camera model:
  - `models/vendor/yolo11/yolov11n_320x320.q.onnx`
- Default visual demo confidence:
  - `0.25`
- Vendor low-latency benchmark model:
  - `models/vendor/yolo11/yolov11n_320x320.q.onnx`
- Vendor320 trustworthy visual runtime:
  - `rt123` = `spacemit-ort.riscv64.1.2.3`
- Vendor320 low-latency benchmark runtime:
  - `rt201` = `spacemit-ort.riscv64.2.0.1`
- Vendor320 `rt201` visual workaround:
  - auto-enabled only for visual helpers when you explicitly force `BANANA_DEMO_RUNTIME_TAG=rt201`
  - guarded by the validated official vendor320 model SHA256 allowlist
  - disable with `BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX=0`
- Board-local camera defaults:
  - `DISPLAY_FLAG=auto`, `HEADLESS_FLAG=auto`, `MAX_FRAMES=0`
- Fast-live camera defaults:
  - vendor320 visual path on `rt123`
  - `320x320` letterbox inference
  - `640x480 @ 60` camera request
- Host-wrapper camera defaults:
  - `DISPLAY_FLAG=0`, `HEADLESS_FLAG=auto`, `MAX_FRAMES=200`
- Board app root after deploy:
  - `/home/svt/banana-yolo11-spacemit-demo`
- Required photo for reproducible image tests:
  - `/home/svt/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg`
- USB camera:
  - default dynamic640 path prefers MJPG at `1280x720`
  - fast-live path intentionally uses the best measured mode for the chosen request, which is currently `YUYV` at `640x480`

## Troubleshooting

- Display over SSH:
  - board-local visual helpers now use `DISPLAY_FLAG=auto` and try to recover a usable display from Wayland/X11 socket hints:
    - `XDG_RUNTIME_DIR=/run/user/<uid>`
    - `WAYLAND_DISPLAY=wayland-0` when present
    - `/run/user/<uid>/.mutter-Xwaylandauth.*` plus `DISPLAY=:0`
  - if GUI still fails, the app now prints an explicit fallback warning and continues headless with periodic frame progress
  - force headless manually with `DISPLAY_FLAG=0`
- Vendor 320x320 detections look wrong or disappear:
  - use `rt123` for trustworthy vendor320 image/camera inference:
    - `BANANA_DEMO_RUNTIME_TAG=rt123`
  - if you explicitly force `rt201` in the visual helpers, the scripts now auto-enable the validated public workaround
  - disable that workaround only when you intentionally want the raw perf stack:
    - `BANANA_DEMO_RUNTIME_TAG=rt201 BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX=0`
  - a clean-room recheck on 2026-03-14 showed that raw `rt201` remains wrong even without any `/dev/tcm` contention
  - `rt202b1` still does not fix vendor320 even with the same public workaround
  - public `1.2.4` is good for vendor320, but it still breaks dynamic640, so it is not the repo default
  - keep the default 640x640 dynamic INT8 path for the best user-facing visual quality
  - use `./scripts/run_camera_demo_fast.sh` when you want the responsive trusted live path instead of the highest-detail live path
- Vendor runtime accidentally replaced by system ORT:
  - the run scripts force `LD_LIBRARY_PATH` to the staged vendor runtime before launching the app
- Reproducibility helper:
  - `./scripts/vendor320_runtime_matrix.sh` saves a compact `rt123` / `rt201 raw` / `rt201 fixed` comparison table plus annotated outputs
  - `./scripts/capture_camera_affinity.sh` saves a compact camera thread-placement and CPU-utilization bundle for the default or fast-live profile
  - `./scripts/check_doxygen_coverage.sh` proves that every tracked source/script/CMake file carries an `@file` block

## Licensing and vendor binaries

This repository does not vendor the SpacemiT runtime tarball. It is fetched by script and remains subject to the vendor's licensing terms.
