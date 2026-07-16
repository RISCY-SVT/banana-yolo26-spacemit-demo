# Full Camera FPS

The selected complete camera surface is `1280x720@60.000 MJPG`, V4L2,
`low-latency`, latest-frame flow, GUI display, boxes, and timing overlay,
without O2. Three independent 180-second runs processed and displayed
5.916864 FPS across 3200 measured frames.
Capture arrival was 9.980414 FPS; the queue-depth-one application
replaced 40.869097% of captured frames because capture was faster than
the complete pipeline.

Mean/p95 software time from `VideoCapture::read` return through display/event
handling was 218.715619 / 262.011456
ms. Mean total software-loop time was 168.738869 ms. These values
include capture, exact letterbox and RGB conversion, executor, deletterboxing,
boxes, overlay, `imshow`, and event handling. They are not pure-model FPS and are
not sensor-to-screen latency because sensor timestamps were not correlated.

The separate MJPG AVI recording arm processed 4.738854
FPS. The 30-minute selected-profile camera soak processed
10679 frames at
5.931789 FPS and completed without an application
failure. OpenCV/V4L2 did not expose an independent driver-drop count, so the report
only claims application replacements.
