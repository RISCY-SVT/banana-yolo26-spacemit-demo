# Custom INT8 Backend Skeleton

This directory is a Stage 0 placeholder for a future YOLO26 custom INT8 backend.
It does not change the default application backend, does not call IME kernels,
and does not implement full inference.

Current allowed path:

- keep model/runtime contracts documented under `stages/`;
- build and test the standalone `custom_int8_engine`;
- add backend wiring only after a later approved stage.
