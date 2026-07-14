# K1X INT8 Executor Limitations

- The package is frozen to one accepted YOLO26n-640 model SHA-256 and one
  integer profile. Arbitrary ONNX models are unsupported.
- Dynamic input shapes, batch sizes other than one, and resolutions other than
  640 are unsupported.
- `run_rgb` expects an already-letterboxed 640x640 RGB buffer. The OpenCV CLI
  performs decode and letterbox outside the pure executor.
- CPU0-3 are the only approved IME cores. CPU4-7 must not execute IME.
- The executor is not thread-safe per handle; use one handle per serialized
  stream.
- Package hashes detect accidental corruption and stale artifacts. They do not
  claim cryptographic authenticity.
- `SCHED_RR` requires privileges and careful lab operation. `SCHED_OTHER` is
  the supported handoff default.
- Legacy float-QDQ output is not the exact arithmetic authority.
- The RGB stem currently uses the exact generic C3-through-K8 dense route.
  Dedicated RGB/RGBX stem kernels remain performance work.
- N4/N8 head convolutions use the exact masked-N16 IME route; they do not yet
  avoid every unused output lane. Grouped/depthwise Conv uses an exact
  four-worker direct scalar arithmetic path rather than a selected RVV kernel.
- The static attention path is complete and exact, but remains a measured
  optimization target together with the stem, small-N head, and grouped Conv.
- This release does not include a camera service, default demo-backend change,
  model training, QAT, a student model, or Q31 promotion.
- Production readiness and 20 FPS are not claimed.
