# Full board ORT baseline

The best safe measured arm is vendor ORT 1.20.2+spacemit, CPU EP,
`ORT_ENABLE_ALL`, sequential execution, inter-op 1, intra-op 4, `taskset -c 0-3`:
`461603.297250 +/- 435.600042 us` (10 warmups, 100 runs, 5 repeats, CV
0.094367%). CPU0-7 vendor-only scouts exited without SIGILL, but six/eight threads
regressed sharply; intra4 on CPU0-3 remains selected.

This is synthetic-input pure ONNX graph latency. It excludes product camera,
decode, preprocessing, rendering, and any production FPS claim.
