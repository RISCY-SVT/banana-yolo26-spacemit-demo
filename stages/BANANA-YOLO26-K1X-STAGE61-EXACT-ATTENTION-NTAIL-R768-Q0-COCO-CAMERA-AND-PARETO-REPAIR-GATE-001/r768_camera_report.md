# R768 Camera Report

Stage61 compares R640 and R768 with the same physical camera, GUI,
latest-frame flow, confidence threshold, CPU/IRQ profile, and recording
disabled. Each matrix row is the mean of three 180-second runs.
The direct probe and OpenCV runs resolve camera `/dev/v4l/by-id/usb-JQ-FAY-220422_BBA_USB_CAMERA_01.00.00-video-index0`.
For the 640x480 source, R768 necessarily upsamples the decoded camera frame;
that arm is a latency/control surface, not evidence of added source detail.

Direct V4L2 values are driver-dequeued compressed-buffer rates. OpenCV
values are decoded-frame rates. Processed/displayed values are application
rates. None is labeled raw sensor FPS, and the read-to-display-call value
is not sensor-to-screen latency.

| Camera mode | Executor R | V4L2 buffers/s | OpenCV decoded/s | Processed/displayed FPS | Replaced slot % | Executor ms | Read-return-to-display-call ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640x480 MJPG | 640 | 30.002 | 14.995 | 6.958 | 53.598 | 130.642 | 176.911 |
| 640x480 MJPG | 768 | 30.002 | 14.998 | 4.571 | 69.521 | 198.637 | 252.161 |
| 1280x720 MJPG | 640 | 30.002 | 9.999 | 6.291 | 37.086 | 130.952 | 209.391 |
| 1280x720 MJPG | 768 | 30.002 | 9.997 | 4.362 | 56.368 | 199.259 | 279.162 |

The direct probes report monotonic/soe (640x480) and monotonic/soe (1280x720) V4L2 timestamps. Sequence gaps are 640x480=0 and 1280x720=0; these driver-visible gaps do not establish unknown upstream sensor or USB losses.

The selected R768 stability control uses 1280x720 MJPG and ran for 1800.1 seconds at 4.338 processed/displayed FPS. The camera result does not change the Q0 COCO selection gates or promote R768 as a default profile.
