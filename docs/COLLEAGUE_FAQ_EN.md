# Colleague FAQ

## What is the full FPS directly from the camera?

Stage59 provides two matched full-GUI presets. `quality-wide` requests
1280x720 MJPG at 60 FPS and processed/displayed 5.976975 FPS across three
independent 180-second runs. `performance` requests 640x480 MJPG at 60 FPS and
processed/displayed 6.619983 FPS under the same protocol, a 10.758% increase.
Both use V4L2, latest-frame flow, `low-latency` without O2, boxes, and the
timing overlay. The OpenCV decoded-frame rates were 9.999262 and 14.993076 FPS;
the corresponding application slot-replacement rates were 40.225839% and
55.846398%. Mean decoded-read-return-to-display-call latency was 219.143 ms for
quality-wide and 184.200 ms for performance.

The separate 30-minute public-launcher soaks measured 5.980417 FPS for
quality-wide and 6.818437 FPS for performance. The performance launcher also
applied its selected reversible CPU5/xHCI IRQ profile; its consumer-loop mean,
p95, and p99 were 146.540, 149.710, and 154.882 ms. These long-soak values are
not pooled with the matched no-profile preset comparison above.

The no-recording number includes capture, exact resize/letterbox, BGR-to-RGB,
the executor, box mapping, boxes, overlay, `imshow`, and event handling. It is
not pure-model FPS. On the performance preset, bounded asynchronous MJPG
recording processed 6.562994 FPS and wrote 6.522417 FPS, within 0.243% of the
matched no-recording control; synchronous recording was rejected at 6.077503
FPS. Its separate 30-minute stability run processed 6.716931 FPS and wrote
6.712772 FPS: 12,322 readable MJPG frames, zero recorder-queue replacements,
and zero write failures. That is 1.489% below the selected 30-minute
no-recording performance surface. Direct V4L2 MMAP telemetry measured
approximately 30.002 dequeued buffers per second with no sequence gaps in each
tested MJPG mode. These kernel
timestamps are monotonic SOE timestamps, but no complete sensor-to-display
timestamp chain exists, so the reported call latency is not sensor-to-screen
latency.

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
/data/y26-k1x-int8-executor/0.9.3/scripts/run_camera_demo.sh
```

The selected camera launchers intentionally do not apply O2. The quality
launcher uses 1280x720 MJPG; `run_camera_demo_fast.sh` uses the measured
640x480 MJPG performance preset and a reversible camera-only CPU5/xHCI IRQ
profile. O2 is available only through the explicitly named
`run_camera_demo_o2_diagnostic.sh`. Pass
`--headless --save-frame /data/Screenshots/yolo26.png`
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
`find_package(y26K1xExecutor 0.9.3 CONFIG REQUIRED)` and link
`y26::executor_shared` or `y26::executor_static`. Initialize options, create one
handle, prepare once with the expected manifest, run serialized frames, and
destroy after all calls finish. Use either float32 NCHW RGB `[0,1]` or an
already-letterboxed interleaved RGB8 640x640 input. Output is 300 float rows of
`[x1,y1,x2,y2,confidence,class]` in letterbox coordinates. The caller maps
boxes back to the original frame; the demo is the complete example.

The executor uses IME only on CPU0-3 and CPU4 as controller. Do not schedule IME
on CPU4-7. This is an optimized engineering handoff, not production
certification and not a 20 FPS claim.
