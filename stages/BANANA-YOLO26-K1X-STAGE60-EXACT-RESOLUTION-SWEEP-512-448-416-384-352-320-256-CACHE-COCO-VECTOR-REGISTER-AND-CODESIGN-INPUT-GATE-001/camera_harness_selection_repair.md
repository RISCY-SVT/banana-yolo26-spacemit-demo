# Camera Harness Selection Repair

## Rejected First Matrix

The first six-arm R512/R384 camera comparison did not export the frozen 0.9.2
operator-selection environment used by the pure-model, COCO, and pipeline
harnesses. It therefore exercised research defaults instead of the accepted
executor route. The mismatch was visible in the executor component:

- R384: about 78.42 ms instead of the matched 47-48 ms class;
- R512: about 150.67 ms instead of the matched 100 ms class.

Those six runs are retained in `camera_unmatched_route_matrix.tsv` and raw
`camera/unmatched-default-route/compare` evidence. They are not used to select
or report a Stage60 camera finalist. The subsequently started unmatched R384
soak was terminated with SIGTERM; its profile restored and its partial output
is retained under `camera/unmatched-default-route/soak`.

## Repair

`stage60_board_camera.sh` now exports the same frozen Stage53-57 operator route
as every accepted Stage60 executor harness. It also records SHA-256 identities
for the demo binary, package manifest, labels, and reversible camera-profile
script for each arm.

A corrected 60-second R384 control measured:

- executor mean: 47.311585 ms;
- OpenCV decoded rate: 15.001661 frames/s;
- processed/displayed rate: 15.001661 frames/s;
- application slot replacements: 0.

This matched the pure-model/RGB evidence closely enough to authorize the full
comparison rerun.

## Selected-Route Matrix

The official matrix uses three 180-second GUI runs per resolution with the same
640x480 MJPG camera mode, latest-frame flow, one OpenCV thread, CPU5 capture,
and reversible camera CPU/xHCI IRQ profile.

Across the three runs:

- R512 averaged 8.821453 processed/displayed frames/s, 14.997412 OpenCV-decoded
  frames/s, 41.180203% application slot replacement, and 97.107686 ms executor
  time.
- R384 averaged 15.000569 processed/displayed frames/s, 15.000569 OpenCV-decoded
  frames/s, zero application slot replacement, and 47.655460 ms executor time.
- The independent V4L2 MMAP probe measured 30.002748 driver-dequeued compressed
  MJPG buffers/s with zero sequence gaps and monotonic SOE timestamps.

R384 proceeds to the camera soak as a latency diagnostic only. Its 6.420383 AP
loss prevents deployment selection regardless of camera throughput.
