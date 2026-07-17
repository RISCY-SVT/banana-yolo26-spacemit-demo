# Camera Preset Decision

The preset comparison used the same physical scene, GUI/display path,
low-latency scheduler, no O2 profile, 30 warm-up frames, three independent
180-second runs, OpenCV thread count one, and reusable preprocessing buffers.

`quality-wide` remains 1280x720 MJPG with 60 FPS requested and reported by the
backend. It averaged 5.976975 processed/displayed FPS, 9.999262 OpenCV-decoded
frames/s, and 40.225839% application queue-slot replacement.

The selected `performance` preset is 640x480 MJPG with 60 FPS requested and
reported. It averaged 6.619983 processed/displayed FPS, 14.993076
OpenCV-decoded frames/s, and 55.846398% application queue-slot replacement.
The matched 640x480 YUYV 30 control averaged 6.604610 processed/displayed FPS.
MJPG therefore wins the measured full-GUI surface narrowly over YUYV and by
10.758% over quality-wide.

The requested/reported 60 FPS value is not a measured sensor rate. The direct
V4L2 evidence reports the dequeued compressed-buffer cadence separately.
Application slot replacement is not a complete camera-drop metric.

Separate 30-minute public-launcher validation passed at 5.980417
processed/displayed FPS for quality-wide and 6.818437 FPS for performance. The
performance launcher includes the independently selected reversible camera IRQ
profile, so those long rows are not pooled with the matched no-profile preset
comparison above.
