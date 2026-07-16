# Colleague FAQ

## What is the full FPS directly from the camera?

The selected Stage58 live surface is 1280x720 at 60 FPS MJPG, V4L2,
latest-frame, `low-latency` without O2, GUI display, boxes, and timing overlay.
Across three independent 180-second runs it processed and displayed 5.573729
FPS. Observed capture arrival was 9.877425 FPS and the application replaced
43.716045% of captured frames because capture was faster than the complete
pipeline. Mean/p95 read-return-to-display software latency was 229.061194 /
271.674422 ms.

The no-recording number includes capture, exact resize/letterbox, BGR-to-RGB,
the executor, box mapping, boxes, overlay, `imshow`, and event handling. It is
not pure-model FPS. Recording is a separate measured surface at 4.706807 FPS.
The 30-minute camera soak passed: 10,527 processed/displayed frames at 5.848 FPS,
with 41.35% application-level latest-frame replacement and no demo failure.
Capture-driver drop accounting was not exposed,
so only application replacements are known. No sensor timestamps were
correlated; the reported latency is not sensor-to-screen latency.

## What is the model resolution?

Exactly 640x640 RGB after aspect-preserving letterbox. The camera may capture a
supported 640x480, 1280x720, or other mode, but the model never receives that
native shape. At confidence 0.25 and IoU 0.50, COCO diagnostic recall first
exceeded 50% in the 64-96 pixel shorter-side bin after letterbox. This is not a
universal minimum: class, contrast, blur, occlusion, lens, distance, and
threshold matter. See `MODEL_RESOLUTION_AND_OBJECT_SIZE_EN.md` and the TSVs.

## Can we see a real Banana camera demo?

Yes. On the BPI-F3:

```bash
/data/y26-k1x-int8-executor/0.9.1/scripts/run_camera_demo.sh
```

The selected full-camera launcher intentionally does not apply O2: on this
board, O2 constrained the capture thread and reduced full-pipeline throughput.
`run_camera_demo_fast.sh` remains an explicit O2 diagnostic for the dedicated
pure-executor policy. Pass `--headless --save-frame /data/Screenshots/yolo26.png`
for a non-GUI run. The release includes
real board screenshots in `outputs/screenshots/` and an annotated recording in
`outputs/demo-video/`. GUI keys are `q`/Esc, `s`, `r`, and Space.

## How do we integrate the library and model?

The runtime model is the complete `package/` directory, not the source ONNX.
Keep its expected manifest SHA-256:

```text
fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be
```

Use `pkg-config --cflags --libs y26-k1x-int8-executor` or
`find_package(y26K1xExecutor 0.9.1 CONFIG REQUIRED)` and link
`y26::executor_shared` or `y26::executor_static`. Initialize options, create one
handle, prepare once with the expected manifest, run serialized frames, and
destroy after all calls finish. Use either float32 NCHW RGB `[0,1]` or an
already-letterboxed interleaved RGB8 640x640 input. Output is 300 float rows of
`[x1,y1,x2,y2,confidence,class]` in letterbox coordinates. The caller maps
boxes back to the original frame; the demo is the complete example.

The executor uses IME only on CPU0-3 and CPU4 as controller. Do not schedule IME
on CPU4-7. This is an optimized engineering handoff, not production
certification and not a 20 FPS claim.
