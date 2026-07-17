# Stage59 Final Report

Classification: `stage59-final-closure-pass-with-camera-scene-limitation`.

Stage59 closes the frozen YOLO26n-640 current-graph release without changing
executor arithmetic, model assets, quantization, layout, or IME placement. The
one remaining physical limitation is the fixed remote camera scene; the full
software, telemetry, archive, exactness, and stability gates pass.

## Release Regression

The published 0.9.1 top-level cross-build omitted
`-mtune=spacemit-x60 -funroll-loops`. That build-contract regression, present
from the first Stage58 commit, explains its approximately 141 ms pure-model
surface. The 0.9.2 build restores the accepted flags without changing any
arithmetic route.

A neutral ABI1 `dlopen` benchmark compared the final library against a
same-session Stage57 build using 1,000 interleaved runs per arm. Stage57
measured 133356.369 us mean; final 0.9.2 measured 134100.921 us mean and
135724.150 us p95. The +0.558% delta passes the required within-1% equivalence
gate and is 5.115% faster than the accepted Stage58 141330.576 us release
mean. The separate 13,500-run 0.9.2 soak measured 133381.666593 us mean,
135151.521 us p99.9, and 135853 us maximum.

## Exactness And API

- ABI1 metadata reads now obey the same busy contract as output and boundary
  reads; run/read races and independent handles pass.
- SIGINT, SIGTERM, and SIGHUP use a signal-safe stop path. Capture and writer
  threads join, media/metrics remain readable, and system profiles restore.
- Official K1X release configuration fails closed without IME, RVV, and the
  frozen profile. The shared library exports 15 ABI1 C symbols and no C++ API.
- F0-F7, bus, Zidane, all 215 boundaries, FRM/vcsr restoration, affinity, and
  CPU4-7 IME-zero checks pass.
- Full COCO completed 5000/5000 with byte-identical prediction SHA-256
  `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`
  and mAP50-95 `0.3707408944391919`.

## Camera Closure

Direct 180-second V4L2 MMAP probes measured 30.0016 dequeued compressed
buffers/s for both selected MJPG modes, with monotonic SOE timestamps and zero
driver-visible sequence gaps. This is distinct from OpenCV decoded rate and
application processed/displayed rate.

The quality preset is 1280x720 MJPG. Its 30-minute full-GUI run measured
9.999235 decoded frames/s, 5.980417 processed/displayed frames/s, 40.191257%
application slot replacement, and 218.780725 ms decoded-read-return-to-display
call time.

The selected performance preset is 640x480 MJPG with the reversible CPU5/xHCI
IRQ profile. Its 30-minute full-GUI run measured 14.983294 decoded frames/s,
6.818437 processed/displayed frames/s, 54.493071% application slot replacement,
and 180.007208 ms decoded-read-return-to-display call time. The metric schema
is version 2 and does not claim raw sensor FPS or sensor-to-screen latency.

Bounded asynchronous MJPG recording on CPU6 measured 6.712772 recorded FPS in
the separate 30-minute run, a 1.489% processed-rate cost versus no recording,
with 12,322 readable frames, zero recorder queue replacements, and zero writer
failures. The optional UI sidecar and direct compressed-buffer backend remain
unselected.

## Distribution And Policy

Release 0.9.2 has separate deterministic runtime and internal-R&D trees. The
runtime delivery contains the SDK and prepared package but no source ONNX. The
internal-R&D delivery adds the exact
`manual_e2e_rep_conv_matmul_qdq.onnx` with SHA-256
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
and internal-only provenance/license records. External ONNX redistribution is
not cleared.

All four final archives are reproducible byte-for-byte, pass clean extraction,
dependency verification, healthcheck, known fixture, image/camera smoke, and
CMake/pkg-config/static/shared consumers on the tested Bianbu 2.2.1 board
image. The dependency verifier was corrected to recognize SONAME symlinks and
then revalidated from all four final archives.

No approved repository license grant was available. The selected factual
policy is `license-decision-pending` plus an internal-use notice; no open-source,
external distribution, production, or certification claim is made.

## Final State

O2 and the camera profile are inactive, IRQ/workqueue state is restored, no
demo process remains, and all large evidence is on NVMe `/data`. Stage59 made
zero eMMC project writes. The unchanged graph remains frozen for maintenance
only. The recommended next action is a human project-license decision; any
new performance, Q31, model, training, or co-design work requires a separately
authorized branch or project.
