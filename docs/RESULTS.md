# Results

This file is updated after board validation.

## YOLO26 FP32 baseline and INT8 closure, 2026-06-30

Run directory:

```text
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
```

YOLO26 FP32 end-to-end 640 is now the working R&D baseline on rt204. PyTorch,
ONNX Runtime CPU, and rt204 SpaceMIT EP agree semantically on the public
standard sanity suite. The suite includes Ultralytics bus/zidane assets,
COCO-derived task-local images, a blank negative control, and the private
canonical photo only as an additional non-public reference.

| Metric class | Runtime | Model | Mean latency ms | FPS | Status |
| --- | --- | --- | ---: | ---: | --- |
| `perf_test forward` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 568.943339 | 1.75761 | pass |
| `app forward-only` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 564.531070 | 1.771382 | pass |
| `app full image benchmark` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 521.868004 | 1.916193 | pass on Ultralytics bus |
| `app full image single` | rt204 SpaceMIT EP | YOLO26n FP32 e2e 640 | 696-715 | 1.40-1.44 | pass on bus, COCO-like, and private canonical images |

YOLO26 INT8 ONNX board acceleration is closed as blocked for the current rt204
path: CPU-good manual Q/DQ INT8 exists, but rt204 SpaceMIT EP fails Q/DQ Conv
compile with `output_type not implemented for clip minmax`. The vendor-ready
minimal repro is documented in `docs/RT204_QDQ_CONV_VENDOR_BUG_REPORT.md`.

See:

- `docs/YOLO26_FP32_BASELINE.md`
- `docs/YOLO26_INT8_STATUS.md`
- `docs/YOLO26_VS_YOLO11_BASELINE.md`
- `docs/CUSTOM_IME_INTEGRATION_FEASIBILITY.md`

For the final consolidated FPS/latency view across production, experimental,
and rejected/P2 variants, see `docs/FPS_SUMMARY.md`. That file is the
authoritative summary table; this file preserves the underlying Day-by-Day
result history.

## Day 3 release packaging and handoff readiness, 2026-06-28

Run directory:

```text
/data/ncnn-logs/ort-logs/2026-06-28_18-57-22/
```

Day 3 preserved the Day 2 production policy:

- Primary production visual branch: generated `dynamic640` INT8 on `rt201`.
- Default normal camera branch: generated `dynamic640` INT8 on `rt201`.
- Fast-live branch: official vendor320 INT8 on `rt123`, `320x320` letterbox.
- Vendor320 raw `rt201` remains benchmark/perf-only.
- FP16 remains experimental; usable coverage remains keep_io 640 on
  `rt201`/`rt202b1`.
- Stable `rt202` remains evaluated but not adopted.
- YOLO26n remains P2.

### Day 3 packaging findings

| Check | Result | Notes |
| --- | --- | --- |
| Day 0 / Day 1 / Day 2 continuity | pass | Required production-week commits are present. |
| `origin/master` baseline | pass | Local Day 2 baseline matched `origin/master` at Day 3 start. |
| Frozen scope consistency | pass | README and production docs agree on the policy above. |
| Current repo build/deploy | pass | `./scripts/build_cross.sh` and `./scripts/deploy_to_banana.sh`. |
| Loader proof | pass | Production binaries resolve repo-local ONNX Runtime, SpaceMIT EP, and staged OpenCV. |
| Release package | pass | Operator-facing artifacts generated under `release/`. |
| Fresh clone runtime fetch | fixed | Raw `+beta1` URL for `rt202b1` produced 404; lock file now uses `%2Bbeta1`. |

### Day 3 smoke results

| Case | Runtime | Result | Key timing / note |
| --- | --- | --- | --- |
| Default image visual | `rt201` | pass | 14 objects, total `843.955 ms`, `1.185 FPS` full image run. |
| Normal camera | `rt201` | pass | Stable `/dev/v4l/by-id/... -> /dev/video20`, MJPG `1280x720`, frame 30 total `242.878 ms`, effective `2.365 FPS`. |
| Fast-live camera | `rt123` | pass | Stable camera path, YUYV `640x480`, frame 60 total `64.167 ms`, effective `10.571 FPS`. |
| Forced headless normal | `rt201` | pass | Explicit headless fallback and progress logs, frame 30 total `249.434 ms`. |
| Forward benchmark smoke | `rt201` vendor320 perf | pass | app forward `23.934424 ms`, `41.780826 FPS`. |
| Full benchmark smoke | `rt123` vendor320 visual | pass | app full `56.165549 ms`, `17.804509 FPS`. |

Day 3 generated annotated examples in the run directory under
`output_examples/day3/`.

## Day 2 RC soak and stable rt202 evaluation, 2026-06-28

Run directory:

```text
/data/ncnn-logs/ort-logs/2026-06-28_17-50-14/
```

Production policy confirmed:

- Primary production visual branch: generated `dynamic640` INT8 on `rt201`.
- Default image visual path: generated `dynamic640` INT8 on `rt201`.
- Default normal camera path: generated `dynamic640` INT8 on `rt201`.
- Fast-live camera path: official vendor320 INT8 on `rt123`, `320x320` letterbox.
- Vendor320 trusted visual path: official vendor320 INT8 on `rt123`.
- Vendor320 low-latency benchmark path: raw `rt201`, perf-only.
- FP16: experimental; `keep_io` FP16 `640x640` remains usable on `rt201` and `rt202b1`.
- Stable `rt202` = `spacemit-ort.riscv64.2.0.2` is fetchable/selectable for
  reproducible testing, but it is **not adopted** for production or FP16
  replacement after Day 2.

### Day 2 stable rt202 decision

| Question | Decision | Evidence |
| --- | --- | --- |
| Can stable `rt202` replace `rt202b1` in active helper/docs? | No | `rt202b1` passes FP16 keep_io 640; stable `rt202` aborts on the same model. |
| Can stable `rt202` replace `rt201` for dynamic640 production? | No | Dynamic640 image and forward/full runs abort on stable `rt202`; `rt201` passes. |
| Can stable `rt202` improve FP16 640 experimental path? | No | Stable `rt202` aborts; `rt201` and `rt202b1` pass. |
| Can stable `rt202` fix vendor320 2.0.x issues? | No | Vendor320 stable `rt202` aborts before a usable semantic verdict. |
| Should beta/RC `rt202b1` be removed from active repo paths? | No | It remains the only validated 2.0.2-line experimental FP16 640 path. |

Stable `rt202` was retested after a board reboot. Post-reboot `/dev/tcm` had no
`fuser`/`lsof` users, but stable `rt202` still aborted on dynamic640, FP16 640,
and vendor320 checks.

### Day 2 image regression

| Path | Runtime | Model/profile | Status | Objects | Preprocess ms | Inference ms | Postprocess ms | Total ms | FPS | Semantic verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default production visual | `rt201` | generated `dynamic640` INT8 | pass | 14 | 28.051 | 791.126 | 17.337 | 858.361 | 1.165 | reasonable |
| Vendor320 trusted visual | `rt123` | official vendor320 INT8 | pass | 8 | 11.615 | 151.690 | 10.349 | 193.223 | 5.175 | reasonable; first run includes startup overhead |
| Vendor320 raw perf sanity | `rt201` | official vendor320 INT8, workaround off | pass as raw/perf | 0 | 10.336 | 141.224 | 4.296 | 166.567 | 6.004 | not a visual default |
| Vendor320 `rt201` workaround | `rt201` | SHA256-guarded visual workaround | pass | 9 | 10.948 | 142.432 | 4.492 | 178.049 | 5.616 | reasonable, non-default |
| Fast-live still path | `rt123` | official vendor320 INT8 | pass | 8 | 10.534 | 63.845 | 4.400 | 98.798 | 10.122 | reasonable |
| FP16 keep_io 640 | `rt201` | experimental FP16 | pass | 13 | 28.166 | 420.125 | 17.430 | 485.721 | 2.059 | reasonable |
| FP16 keep_io 640 | `rt202b1` | experimental FP16 | pass | 13 | 27.989 | 438.632 | 17.298 | 503.659 | 1.985 | reasonable |
| FP16 keep_io 320 | `rt201` | experimental FP16 | expected fail | n/a | n/a | n/a | n/a | n/a | n/a | EP reshape failure |
| Dynamic640 candidate | `rt202` stable | generated `dynamic640` INT8 | fail | n/a | n/a | n/a | n/a | n/a | n/a | aborts with `std::runtime_error` |
| FP16 keep_io 640 candidate | `rt202` stable | experimental FP16 | fail | n/a | n/a | n/a | n/a | n/a | n/a | aborts with `std::runtime_error` |
| Vendor320 candidate | `rt202` stable | official vendor320 INT8 | fail | n/a | n/a | n/a | n/a | n/a | n/a | aborts; no modern-runtime vendor320 fix |

### Day 2 camera regression

| Case | Runtime | Display/headless | Camera mode | Frame | Total ms | FPS | Effective FPS | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Normal camera default | `rt201` | display auto -> on, headless off | `/dev/v4l/by-id/... -> /dev/video20`, MJPG, `1280x720` | 60 | 245.835 | 4.068 | 2.316 | pass |
| Fast-live camera | `rt123` | display auto -> on, headless off | `/dev/v4l/by-id/... -> /dev/video20`, YUYV, `640x480` | 80 | 69.089 | 14.474 | 8.928 | pass |
| Forced headless normal | `rt201` | display off, headless on | same USB camera | 20 | 248.222 | 4.029 | 2.279 | pass |
| Forced headless fast-live | `rt123` | display off, headless on | same USB camera | 50 | 66.061 | 15.137 | 10.390 | pass |
| Normal still save | `rt201` | headless still | JPEG `1280x720` | 1 | 863.934 | 1.157 | 0.758 | pass |
| Fast-live still save | `rt123` | headless still | JPEG `640x480` | 1 | 88.895 | 11.249 | 2.356 | pass |

Stable `rt202` was not expanded into camera testing because it failed the image
and performance gates first.

### Day 2 performance regression

Benchmark settings for app helper runs: `BENCH_PERF_REPEATS=50`,
`BENCH_WARMUP=5`, `BENCH_RUNS=20`, `BENCH_REPEATS=3`, `threads=4`, `pin=cluster0`.

| Case | Runtime | Model/profile | perf_test mean ms | perf_test FPS | App mode | App mean ms | App FPS | Notes |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Vendor320 low-latency perf | `rt201` | official vendor320 INT8 | 24.4143 | 40.9483 | forward | 24.210065 | 41.305136 | perf-only branch |
| Vendor320 trusted visual control | `rt123` | official vendor320 INT8 | 48.3266 | 20.6767 | forward | 49.095494 | 20.368468 | visual branch |
| Vendor320 trusted visual control | `rt123` | official vendor320 INT8 | n/a | n/a | full | 57.540777 | 17.378980 | full image pipeline |
| Primary dynamic640 visual | `rt201` | generated dynamic640 INT8 | 190.024 | 5.2623 | forward | 190.567794 | 5.247476 | primary production visual branch |
| Primary dynamic640 visual | `rt201` | generated dynamic640 INT8 | n/a | n/a | full | 233.480423 | 4.283014 | full image pipeline |
| FP16 keep_io 640 | `rt201` | experimental FP16 | 270.867 | 3.69176 | forward | 273.265611 | 3.659443 | experimental |
| FP16 keep_io 640 | `rt202b1` | experimental FP16 | 293.060863 | 3.41212 | forward | 294.988206 | 3.389966 | experimental |
| Dynamic640 candidate | `rt202` stable | generated dynamic640 INT8 | fail | fail | forward/full | fail | fail | `tcm buffer acquire/alloc failed` / abort |
| FP16 keep_io 640 candidate | `rt202` stable | experimental FP16 | fail | fail | forward | fail | fail | abort; does not replace `rt202b1` |
| Vendor320 candidate | `rt202` stable | official vendor320 INT8 | fail | fail | forward | fail | fail | abort; no vendor320 fix |

### Runtime status as of Day 2

| Runtime | Production status | Notes |
| --- | --- | --- |
| `rt123` | production trusted for vendor320 visual and fast-live | Keep. |
| `rt201` | production primary dynamic640 and vendor320 perf | Keep. |
| `rt202b1` | active experimental FP16 640 fallback | Keep because stable `rt202` failed replacement testing. |
| `rt202` stable | evaluated, not adopted | Pinned and selectable only for reproducibility; crashes current paths. |

YOLO26n remains P2. Day 2 did not reopen YOLO26n R&D because stable `rt202`
failed its runtime gate.

## Day 1 production regression, 2026-06-28

Run directory:

```text
/data/ncnn-logs/ort-logs/2026-06-28_16-43-04/
```

Production policy confirmed:

- Primary production visual branch: generated `dynamic640` INT8 on `rt201`.
- Default image visual path: generated `dynamic640` INT8 on `rt201`.
- Default normal camera path: generated `dynamic640` INT8 on `rt201`.
- Fast-live camera path: official vendor320 INT8 on `rt123`, `320x320` letterbox.
- Vendor320 trusted visual path: official vendor320 INT8 on `rt123`.
- Vendor320 low-latency benchmark path: raw `rt201`, perf-only.
- FP16: experimental; `keep_io` FP16 `640x640` works on `rt201` and `rt202b1`.

### Day 1 image regression

| Path | Runtime | Model/profile | Status | Objects | Preprocess ms | Inference ms | Postprocess ms | Total ms | FPS | Semantic verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default production visual | `rt201` | generated `dynamic640` INT8 | pass | 14 | 27.932 | 820.408 | 17.365 | 887.258 | 1.127 | reasonable |
| Vendor320 trusted visual | `rt123` | official vendor320 INT8 | pass | 8 | 10.326 | 63.708 | 4.411 | 97.765 | 10.229 | reasonable |
| Vendor320 `rt201` raw perf | `rt201` | official vendor320 INT8 | pass | n/a | n/a | 25.135 forward-only | n/a | n/a | 39.786 forward-only | perf-only, not visual default |
| Vendor320 `rt201` workaround | `rt201` | SHA256-guarded visual workaround | pass | 9 | 9.862 | 184.588 | 4.456 | 219.678 | 4.552 | reasonable, slower than `rt123` |
| FP16 keep_io 640 | `rt201` | experimental FP16 | pass | 13 | 29.619 | 406.901 | 17.301 | 474.652 | 2.107 | reasonable |
| FP16 keep_io 640 | `rt202b1` | experimental FP16 | pass | 13 | 27.777 | 435.406 | 17.336 | 499.653 | 2.001 | reasonable |
| FP16 keep_io 320 | `rt201`/`rt202b1` | experimental FP16 | expected fail | n/a | n/a | n/a | n/a | n/a | n/a | remains unsupported |

### Day 1 camera regression

| Case | Runtime | Display/headless | Camera mode | Frame | Total ms | FPS | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Normal camera default | `rt201` | display off, headless on | `/dev/v4l/by-id/... -> /dev/video20`, MJPG, `1280x720` | 60 | 268.892 | 3.719 | pass |
| Fast-live camera | `rt123` | display on, headless off | `/dev/v4l/by-id/... -> /dev/video20`, `640x480` | 80 | 68.883 | 14.517 | pass |
| Forced headless normal | `rt201` | display off, headless on | same USB camera | 20 | 250.104 | 3.998 | pass |
| Forced headless fast-live | `rt123` | display off, headless on | same USB camera | 50 | 67.980 | 14.710 | pass |
| Normal still save | `rt201` | headless still | JPEG `1280x720` | 1 | 863.969 | 1.157 | pass |
| Fast-live still save | `rt123` | headless still | JPEG `640x480` | 1 | 86.837 | 11.516 | pass |

Notes:

- Camera auto-selection still prefers the stable `/dev/v4l/by-id` path.
- No AVI output is created unless recording is explicitly requested.
- Host-wrapper `SAVE_OUTPUT`/positional image and camera still paths were fixed during Day 1 so requested local artifacts are created predictably.

### Day 1 performance regression

| Case | Runtime | Model/profile | perf_test mean ms | perf_test FPS | App mode | App mean ms | App FPS | Notes |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Vendor320 low-latency perf | `rt201` | official vendor320 INT8 | 24.6879 | 40.4957 | forward | 25.134561 | 39.785855 | perf-only branch |
| Vendor320 trusted visual control | `rt123` | official vendor320 INT8 | 48.7262 | 20.5072 | forward | 49.203386 | 20.323804 | visual branch |
| Primary dynamic640 visual | `rt201` | generated dynamic640 INT8 | 190.245 | 5.25614 | forward | 192.232816 | 5.202025 | primary production visual branch |
| Primary dynamic640 visual | `rt201` | generated dynamic640 INT8 | n/a | n/a | full | 232.189709 | 4.306823 | full image pipeline |
| FP16 keep_io 640 | `rt201` | experimental FP16 | 272.964 | 3.663487 | forward | 273.233840 | 3.659869 | experimental |
| FP16 keep_io 640 | `rt202b1` | experimental FP16 | 292.900412 | 3.414130 | forward | 294.169391 | 3.399402 | experimental |

### Day 1 YOLO26n feasibility gate

Result: not adopted; P2 follow-up only.

- Public checkpoint was reachable from `https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt`.
- Checkpoint SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- ONNX 640 export succeeded with Ultralytics `8.3.233`; output contract is traditional YOLO `[1,84,8400]`, not E2E `[1,300,6]`.
- Direct dynamic `xquant` produced an INT8 ONNX artifact.
- Board rt201 app smoke ran both float ONNX and dynamic INT8 ONNX, but both produced a giant false `refrigerator` box on the canonical photo.
- Dynamic INT8 timing was about `1086.868 ms` inference / `1147.946 ms` total for one image; float ONNX timing was about `544.356 ms` inference / `600.287 ms` total.

Decision: YOLO26n is a candidate for separate post-release decode/contract and quantization work, not a Day 1 production branch.

- Image demo
  - Required image: `/home/svt/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg`
  - Vendor INT8 320x320 is now validated as a trustworthy visual path on runtime `rt123` (`spacemit-ort.riscv64.1.2.3`) with letterbox preprocessing.
  - Vendor INT8 320x320 remains the low-latency benchmark path on runtime `rt201` (`spacemit-ort.riscv64.2.0.1`).
  - Public `rt201` also has a correctness workaround for vendor320 visual inference:
    - `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE=1`
    - `SPACEMIT_EP_DISABLE_OP_NAME_FILTER=/model.23/Slice;/model.23/Slice_1;/model.23/Add_1;/model.23/Add_2;/model.23/Sub;/model.23/Sub_1`
    - it produces semantically good canonical-photo detections, but it is much slower than `rt123`
  - The default visual demo path remains the custom dynamic INT8 640x640 path because it is still the best user-facing quality path.
  - A cleaner 640 sample was captured at `--conf 0.25`.

- Camera demo
  - Default camera auto-selection resolves the USB camera through `/dev/v4l/by-id/... -> /dev/video20`.
  - The default camera auto mode chooses MJPG for `1280x720` because it offers `60 FPS` on the connected USB camera, versus `7.5 FPS` for YUYV.
  - A dedicated fast-live path now exists:
    - script: `./scripts/run_camera_demo_fast.sh`
    - model: official vendor320 INT8 ONNX
    - runtime: `rt123`
    - preprocess: letterbox
    - default camera request: `640x480 @ 60`
  - Board-local `run_camera_demo.sh` now defaults to `DISPLAY_FLAG=auto`, `HEADLESS_FLAG=auto`, and `MAX_FRAMES=0`.
  - In a tty shell with no exported GUI vars, the helper detects local Wayland/X11 sockets and attempts live display automatically.
  - If GUI is still unavailable, the helper prints an explicit fallback message and the app emits early plus periodic progress logs instead of appearing stuck.
  - The app now reports:
    - `camera_open_method`
    - `camera_backend`
    - `capture_ms`
    - `affinity_policy`
    - `main_tid`
    so live camera behavior is diagnosable without rebuilding.
  - In display mode, the first raw frame is shown immediately with a warmup banner while the first inference initializes.
  - Headless live inference is stable.
  - Default camera runs no longer create AVI output unless explicitly requested.
  - Camera mode can now save a single annotated still image when `SAVE_OUTPUT` points to an image suffix such as `.jpg`.
  - A board-local camera run from repo root reached:
    - `display_resolved=1`
    - `display_reason=gui-socket`
    - `camera_open_method=index-v4l2`
    - `camera_backend=V4L2`
    - `warmup preview shown while inference initializes`
    - `display active, live preview should now be visible; press ESC/q to exit`
  - A forced headless run reached:
    - `display_resolved=0`
    - `camera running in headless mode; periodic progress logs enabled`
    - early frame logs such as `frame=1 ...` without waiting for 10 frames
  - Measured live camera timing on the default visual path (`dynamic640`, `MAX_FRAMES=20`, headless):
    - frame 1:
      - `capture_ms=355.168`
      - `inference_ms=887.049`
      - `total_ms=943.833`
    - frame 10:
      - `capture_ms=49.250`
      - `inference_ms=212.994`
      - `total_ms=260.773`
    - frame 20:
      - `capture_ms=50.477`
      - `inference_ms=212.787`
      - `total_ms=259.410`
    - whole 20-frame loop:
      - `effective_fps=2.220142`
  - Measured live camera timing on the fast-live path (`vendor320 rt123`, `640x480`, `MAX_FRAMES=20`, headless):
    - frame 1:
      - `capture_ms=327.421`
      - `inference_ms=67.927`
      - `total_ms=89.945`
    - frame 10:
      - `capture_ms=1.176`
      - `inference_ms=50.442`
      - `total_ms=67.134`
    - frame 20:
      - `capture_ms=1.076`
      - `inference_ms=52.964`
      - `total_ms=69.465`
    - whole 20-frame loop:
      - `effective_fps=8.960986`
  - Fast-live candidate comparison:
    - `640x480` request:
      - actual camera mode: `640x480`, `YUYV`, `30 FPS`
      - steady preview rate after warmup: about `14.4-14.9 FPS`
    - `1280x720` request:
      - actual camera mode: `1280x720`, `YUYV`, `7.5 FPS`
      - `capture_ms ~= 832-837`
      - whole 20-frame loop: `effective_fps=0.998994`
    - result:
      - `640x480` is the correct fast-live default on the current USB camera
  - Interpretation:
    - sustained live throughput is now explainable instead of opaque
    - after warmup, the default visual path is limited mostly by `dynamic640` inference cost plus about `50 ms` of camera capture/MJPG decode overhead
    - a low-risk latency tweak (`CAP_PROP_BUFFERSIZE=1`) remains applied
    - the new fast-live profile is the measured low-risk product answer for better responsiveness without abandoning the trusted visual path
  - A focused affinity pass reproduced the earlier cluster0 underutilization report:
    - all demo-side threads initially inherited the same `cluster0` mask
    - GTK helper threads and camera-side work were therefore competing on the same cluster as ORT workers
    - a dedicated split implementation was prototyped and verified with per-thread affinity dumps
    - it did move helper work to `cluster1`, but it did not improve measured live FPS on the current camera/display stack
    - the repo therefore keeps the simpler single-mask policy by default and exposes a regression helper instead of enabling an unhelpful split:
      - `scripts/capture_camera_affinity.sh`

- Forward-only benchmark
  - Vendor320 perf stack (`rt201`) `onnxruntime_perf_test`:
    - `25.4561 ms`
    - `39.2748 FPS`
  - Vendor320 perf stack (`rt201`) application forward-only:
    - `25.821000 ms`
    - `38.728167 FPS`
  - Vendor320 visual stack (`rt123`) `onnxruntime_perf_test`:
    - `49.4587 ms`
    - `20.2038 FPS`
  - Vendor320 visual stack (`rt123`) application forward-only:
    - `50.248301 ms`
    - `19.901171 FPS`
  - Custom dynamic INT8 640x640 `onnxruntime_perf_test`:
    - `201.162 ms`
    - `4.97096 FPS`
  - Custom dynamic INT8 640x640 application forward-only:
    - `203.832438 ms`
    - `4.905990 FPS`

- Full pipeline benchmark
  - Application full pipeline includes preprocess + inference + postprocess.
  - Vendor320 visual stack (`rt123`) application full pipeline:
    - `60.926593 ms`
    - `16.413194 FPS`
  - Application full pipeline 640x640 dynamic INT8:
    - `241.812077 ms`
    - `4.135443 FPS`
  - A corrected vendor320 camera run on the real USB camera with default settings:
    - auto-selected `/dev/v4l/by-id/... -> /dev/video20`
    - auto-selected `MJPG`
    - produced `objects=0` at frame 10 and frame 20 on the captured white-wall scene
    - did not create any new AVI output unless explicitly requested

- Remediation notes
  - The decisive root-cause pass showed that vendor320 is runtime-version-sensitive on the public stack:
    - public vendor example + `1.2.2` = semantically good
    - public vendor example + `1.2.3` = semantically good
    - public vendor example + `2.0.1` = semantically poor
  - A clean-room follow-up pass closed the remaining rt201 question:
    - `/dev/tcm` was clean
    - no `alloc failed(...)` appeared
    - official vendor example on `rt201` still failed semantically
    - repo app on `rt201` still failed semantically
    - `rt202b1` (`2.0.2+beta1`) matched the same bad vendor320 output family as `rt201`
    - public package line `1.2.4` was semantically good for vendor320 but still broke dynamic640 badly
  - A chain-complete follow-up then closed the remaining public 2.0.x questions:
    - official public example path was retested with the exact same `q.onnx` model bundle, exact same canonical photo input hash, and both public decode interpretations (`centerwh` and `xyxy`)
    - `rt201`, tarball `rt202b1`, and package `pkg202fix` all stayed bad for vendor320 across graph optimization levels `0`, `1`, `2`, and `99`
    - public float `yolov11n_320x320.onnx` was not a rescue path on 2.0.x either; the EP failed during reshape/compile instead
    - no compatible public 2.0.x vendor320 chain was found
  - A final EP/runtime-localization pass refined that result:
    - `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE=1` alone does not fix vendor320 on `rt201` or `rt202b1`
    - optimized models generated by `rt123`, `rt201`, and `rt202b1` all replay with the same good CPU semantics on stable `rt123` CPU, so host graph rewrite is not the corruption point
    - the public `rt201` corruption is in EP execution, specifically the `/model.23` tail
    - keeping `/model.23/Slice`, `/model.23/Slice_1`, `/model.23/Add_1`, `/model.23/Add_2`, `/model.23/Sub`, and `/model.23/Sub_1` on CPU, together with `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE=1`, restores a semantically good vendor320 result on `rt201`
    - the same workaround does not restore `rt202b1`
  - Raw output stability split is now explicit:
    - `1.2.x` vendor320 line (`1.2.3`, `1.2.4`) is stable across warmup counts
    - `2.0.x` public line (`2.0.1`, `2.0.2+beta1`) changes vendor320 raw output hashes across warmup counts even when input hash is fixed
  - The repository now encodes that matrix explicitly instead of pretending one tarball is correct for every path:
    - vendor320 visual path -> `rt123`
    - vendor320 low-latency benchmark path -> `rt201`
    - dynamic640 path -> `rt201`
    - forced `rt201` visual runs now auto-apply the validated public workaround, but defaults still prefer `rt123` because the workaround is much slower
  - Demo defaults still use sane auto camera selection, sane auto MJPG selection, and no AVI recording unless explicitly requested.
  - The validated `vendor320 rt201` visual workaround is now guarded by the official vendor320 model SHA256, so it is not silently applied to arbitrary `320`-named models.
  - A compact reproducibility helper now exists:
    - `scripts/vendor320_runtime_matrix.sh`
    - it saves `rt123`, `rt201 raw`, and `rt201 fixed` image outputs plus a compact matrix table
  - A local docs helper now exists:
    - `scripts/gen_doxygen.sh`
    - it generates HTML docs and warning logs even on hosts that do not have `doxygen` installed globally
  - A file-coverage helper now exists:
    - `scripts/check_doxygen_coverage.sh`
    - it proves that every tracked source/header/script/CMake file carries an `@file` block

- Quantization notes
  - Official vendor 640x640 INT8 YOLO11n model was not found in the pinned public archive.
  - Public xquant static calibration for 640x640 was attempted, but the tool still entered a `Runtime Calibration(BlockWise) ... /50` path despite a smaller requested calibration budget.
  - For this repository, the practical 640x640 fallback is `xquant` dynamic INT8.

## FP16 matrix

- Scope:
  - model family: YOLO11n ONNX
  - sizes: `320x320`, `640x640`
  - runtimes tested: `rt123`, `rt201`, `rt202b1`
- Public vendor FP16 artifacts:
  - no public official/vendor YOLO11n FP16 ONNX for `320` or `640` was found in the pinned public sources used by this repo
- Final repo-managed FP16 chain:
  - source float32 baselines:
    - vendor `320x320` float ONNX
    - repo-generated `640x640` float ONNX
  - deterministic conversion helper:
    - `./scripts/fetch_or_build_fp16_models.sh`
  - validated model families:
    - true FP16 I/O: `*.fp16.onnx`
    - internal FP16 with FP32 I/O: `*.fp16_iop32.onnx`
  - the benchmark/result matrix below uses the `keep_io` models because the true FP16 I/O models were not runnable on the tested board runtime lines
- Ultralytics `half=True` note:
  - CPU export with `half=True` was tested and rejected because the produced ONNX remained FP32

### FP16 model provenance

| model | classification | input dtype | output dtype | sha256 | note |
| --- | --- | --- | --- | --- | --- |
| `yolov11n_320x320.fp16.onnx` | true FP16 I/O | `FLOAT16` | `FLOAT16` | `ebcbf994e3fea1ca6dfc9be1a3b7c18ec760b078f71e929c192a0d48c671e411` | dtype-correct, but not runnable on the tested public board runtimes |
| `yolov11n_640x640.fp16.onnx` | true FP16 I/O | `FLOAT16` | `FLOAT16` | `ec7a8b5db9b4b022f98ea3ce7e887119a996637ba07be839b51b4970c2df61fd` | dtype-correct, but not runnable on the tested public board runtimes |
| `yolov11n_320x320.fp16_iop32.onnx` | internal FP16 / FP32 I/O | `FLOAT` | `FLOAT` | `3291474d7a8e40bc0fabf6feb054942675f562dab0c04666bddd47662eb27b69` | final tested `320` FP16 path |
| `yolov11n_640x640.fp16_iop32.onnx` | internal FP16 / FP32 I/O | `FLOAT` | `FLOAT` | `4742625978c4b5cc25282bf02890837fcea7762d5536fe55e583311ce9b14593` | final tested `640` FP16 path |

Dtype validation was done by inspecting ONNX graph input/output types plus initializer dtype counts after conversion.
The pass artifacts also include exported provenance and dtype-summary tables for later audit.

### FP16 perf_test summary

| runtime | model | status | mean ms | FPS | note |
| --- | --- | --- | --- | --- | --- |
| `rt123` | `fp16-320 keep_io` | fail |  |  | heap corruption (`free(): invalid next size`) |
| `rt123` | `fp16-640 keep_io` | fail |  |  | heap corruption (`free(): corrupted unsorted chunks`) |
| `rt201` | `fp16-320 keep_io` | fail |  |  | SpaceMIT EP reshape error |
| `rt201` | `fp16-640 keep_io` | ok | `271.549` | `3.682577` | stable |
| `rt202b1` | `fp16-320 keep_io` | fail |  |  | SpaceMIT EP compile/reshape error |
| `rt202b1` | `fp16-640 keep_io` | ok | `293.624088` | `3.405715` | stable |

### FP16 app forward-only summary

| runtime | model | status | mean ms | std ms | FPS | note |
| --- | --- | --- | --- | --- | --- | --- |
| `rt123` | `fp16-320 keep_io` | fail |  |  |  | heap corruption |
| `rt123` | `fp16-640 keep_io` | fail |  |  |  | heap corruption |
| `rt201` | `fp16-320 keep_io` | fail |  |  |  | metadata/runtime error before usable inference |
| `rt201` | `fp16-640 keep_io` | ok | `273.951889` | `0.548755` | `3.650276` | stable |
| `rt202b1` | `fp16-320 keep_io` | fail |  |  |  | metadata/runtime error before usable inference |
| `rt202b1` | `fp16-640 keep_io` | ok | `295.421328` | `0.616934` | `3.384996` | stable |

### FP16 app full-pipeline summary

| runtime | model | status | objects | preprocess ms | inference ms | postprocess ms | total ms | FPS | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rt123` | `fp16-320 keep_io` | fail |  |  |  |  |  |  | heap corruption |
| `rt123` | `fp16-640 keep_io` | fail |  |  |  |  |  |  | heap corruption |
| `rt201` | `fp16-320 keep_io` | fail |  |  |  |  |  |  | metadata/runtime error |
| `rt201` | `fp16-640 keep_io` | ok | `13` | `28.385` | `431.284` | `17.391` | `497.471` | `2.010` | semantically reasonable canonical-photo output |
| `rt202b1` | `fp16-320 keep_io` | fail |  |  |  |  |  |  | metadata/runtime error |
| `rt202b1` | `fp16-640 keep_io` | ok | `13` | `28.189` | `437.294` | `17.130` | `501.964` | `1.992` | semantically reasonable canonical-photo output |

### FP16 visual sanity and recommendations

- `rt201 + fp16-640 keep_io`
  - recommended only as an experimental FP16 coverage path, not as the default product visual path
  - canonical photo output is semantically reasonable
- `rt202b1 + fp16-640 keep_io`
  - also usable as experimental FP16 coverage
  - slightly slower than `rt201`
- `rt123 + fp16-320/640 keep_io`
  - not recommended
  - observed heap corruption in both perf_test and app paths
- `rt201/rt202b1 + fp16-320 keep_io`
  - not recommended
  - both fail during EP reshape/compile
- true FP16 I/O models (`*.fp16.onnx`)
  - dtype-correct, but not currently a supported board runtime path on these public vendor lines

Practical conclusion:

- this repository now has honest FP16 coverage for YOLO11n, but not a full “everything works” matrix
- the only validated board-side FP16 execution path from this pass is:
  - `keep_io` FP16
  - `640x640`
  - `rt201` or `rt202b1`
- no tested public runtime line produced a usable `320x320` FP16 board result
