# Runtime Loader Report

## Host loader

`ldd` resolves `libonnxruntime.so.1` to:

`/data/banana-yolo26-spacemit-demo/.deps/custom_int8_engine/venv-onnx-stage3/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1`

The backing `1.27.0` library SHA-256 is `b5c9d4f124d24707f514dad926dc181820807178855df1c528e3addb2dd0e6f7`. The executable needs `VERS_1.27.0`.

## Board loader

With `LD_LIBRARY_PATH=/home/svt/spacemit-ort.riscv64.2.0.1/lib`, `ldd` resolves:

`libonnxruntime.so.1 => /home/svt/spacemit-ort.riscv64.2.0.1/lib/libonnxruntime.so.1`

The library SHA-256 is `5a28c8128a7b1ed9cb29357f42eb7a2a45eb1b23d8791c2fee1eaf0151546238`, and the executable needs `VERS_1.20.2`. No accidental system ORT was resolved.

Board loader check: pass. Deployment-policy check: partial because the binary embeds the build-tree RPATH; controlled environment loading was required.
