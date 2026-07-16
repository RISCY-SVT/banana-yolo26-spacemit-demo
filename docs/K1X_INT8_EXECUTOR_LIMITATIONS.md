# K1X INT8 Executor Limitations

- The package is frozen to one accepted YOLO26n-640 model SHA-256 and one
  integer profile. Arbitrary ONNX models are unsupported.
- Dynamic input shapes, batch sizes other than one, and resolutions other than
  640 are unsupported.
- `run_rgb` expects an already-letterboxed 640x640 RGB buffer. The OpenCV demo
  performs decode and letterbox outside the pure executor.
- CPU0-3 are the only approved IME cores. CPU4-7 must not execute IME.
- The executor is not thread-safe per handle; use one handle per serialized
  stream.
- Package hashes detect accidental corruption and stale artifacts. They do not
  claim cryptographic authenticity.
- `SCHED_RR` requires privileges and careful lab operation. `SCHED_OTHER` is
  the supported policy. The opt-in Stage56 frame-gated epoch-spin mode consumes
  more process CPU during inference than condition-variable compatibility, but
  parks workers between frames. It remains optimized research, not a
  general-purpose default.
- Legacy float-QDQ output is not the exact arithmetic authority.
- The dedicated compact-C3 RGB stem, direct 1x1, P3 stride-2, N4/N8 kernels,
  RVV depthwise path, and direct
  attention transforms are specific to the frozen shapes. They are not generic
  operator implementations for arbitrary models.
- Some low-impact Split/Reshape/Concat surfaces retain measured direct or
  reference implementations. Stage55 corrected the legal indexed register and
  vtype contracts and selected exact indexed RVV LUT2 and attention exp lookup.
- The Stage56 O2 isolation profile is reversible and intended only for a
  dedicated board. It changes runtime cgroup, IRQ, workqueue, and service
  placement; it does not select a new boot profile, realtime scheduling, THP,
  eMMC runtime, or alternate kernel.
- Cost-model V4 predicts current-graph composition and measured candidate
  composition within the Stage56 gates. Held-out novel shapes retain high
  worst-case error and require direct measurement.
- This release includes a user-launched camera demo but no installed camera
  service. It does not include model training, QAT, a student model, or Q31.
- Production readiness and 20 FPS are not claimed.
