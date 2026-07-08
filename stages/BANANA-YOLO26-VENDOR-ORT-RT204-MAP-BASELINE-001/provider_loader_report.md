# provider_loader_report

Provider path: C++ SpaceMIT EP via `Ort::SessionOptionsSpaceMITEnvInit`, provider `spacemit`, taskset CPU0-3, 4 rt204 intra-op threads.

The deployment `ldd` resolved `libonnxruntime.so.1` and `libspacemit_ep.so.2` from `/home/svt/banana-yolo26-trackb-rt204-map/runtime/rt204/lib`; no accidental system ORT was used for the predictor binary.

A first `perf_test` attempt without explicit `LD_LIBRARY_PATH` failed by resolving `/lib/libonnxruntime.so.1`; the accepted perf rows were rerun with `LD_LIBRARY_PATH=$BOARD_DIR/runtime/rt204/lib:$BOARD_DIR/opencv/lib`.

Deployment/log proof is stored at:

```text
/data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/run_logs/deploy_trackb_rt204.log
/data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/run_logs/board_stable_fps_matrix_retry_loader.log
```
