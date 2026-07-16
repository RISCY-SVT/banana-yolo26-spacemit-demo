# YOLO26 Camera Demo

The `y26_k1x_demo` application uses the frozen `K1X_INT8_V1` executor directly.
It does not load ONNX Runtime or a vendor execution provider. The model tensor is
always a 640x640 RGB letterbox, regardless of camera capture resolution.

## Start

On the board:

```bash
export RELEASE=/data/y26-k1x-int8-executor/0.9.1
$RELEASE/bin/y26_k1x_demo \
  --package "$RELEASE/package" \
  --expected-manifest-sha256 fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  --labels "$RELEASE/labels/coco80.txt" \
  --source camera:auto --camera-width 1280 --camera-height 720 \
  --camera-fps 60 --camera-fourcc MJPG \
  --profile low-latency --flow latest-frame
```

From the build host, `scripts/run_camera_demo.sh` runs the same command over
SSH. `scripts/run_camera_demo_fast.sh` wraps it in the reversible O2 system
profile. The fast script does not select a different model or resolution.

On the Stage58 board/camera combination, the selected full-camera profile is
the plain `low-latency` launcher without O2. Applying O2 to the whole demo
process prevents the latest-frame capture thread from moving to CPU5-7 and was
slower end to end. O2 remains the selected dedicated pure-executor profile; it
is retained as an explicit diagnostic option, not presented as the
camera-throughput winner.

## Controls

| Key | Action |
|---|---|
| `q` or `Esc` | Exit |
| `s` | Save an annotated PNG under `/data/Screenshots/` |
| `r` | Start or stop MJPG AVI recording |
| Space | Pause or resume display without preparing a new executor |

Use `--headless --save-frame /data/Screenshots/output.png` without a desktop.
Use `--record /data/video/yolo26.avi` for startup recording. If the OpenCV
writer cannot open MJPG AVI, the demo records a timestamped PNG sequence and
reports that fallback.

## Frame Policies

`sequential` processes every frame returned by the capture loop and does not
intentionally drop frames. `latest-frame` uses a capture thread and a queue of
depth one; an old queued frame is replaced when inference is behind capture.
The summary reports replacements as `dropped_frames` and `drop_pct`.

The effective V4L2 mode must be checked with `scripts/detect_camera_formats.sh`.
The requested 60 FPS property is not treated as measured camera arrival rate.

## Timing

The overlay and TSV keep capture, resize/letterbox, BGR-to-RGB, executor,
deletterbox/filter, drawing, display, recording, and loop time separate.
`processed_fps` is the full software-loop rate. `capture_fps` is observed frame
arrival in latest-frame mode. Pure executor FPS is never displayed as camera
FPS. Read-return-to-display time is software latency, not sensor-to-screen
latency because this camera path does not correlate V4L2 sensor timestamps.

For an operating-envelope study, `--detections-tsv FILE` writes buffered
per-detection class, confidence, original-frame box, and 640-letterbox box
coordinates after the run. It does not stream evidence writes during timing.

The measured Stage58 headline and its exact camera mode are in
`COLLEAGUE_FAQ_EN.md` and `PERFORMANCE_AND_ACCURACY.md`.
