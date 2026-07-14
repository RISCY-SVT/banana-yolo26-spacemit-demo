# K1X INT8 Executor Architecture

The executor is a static AOT runtime for one frozen YOLO26n-640 profile.

- One global 64-byte-aligned activation arena uses lifetime-based offsets. The
  resident model4-final through model9 facade binds directly to this arena and
  to the FullExecutor worker pool; it has no copy-in/copy-out arena or second
  pool in headline execution.
- Feature tensors use `NCHWc8_SPATIAL_INNER_V1`; attention/head tensors use a
  fixed row-major integer layout where required.
- Immutable packed weights and arithmetic assets are loaded during prepare.
- Four persistent workers are pinned to CPU0-3. CPU4 is controller-only.
- Dense Conv uses shape-dispatched direct-strided 1x1, P3 stride-2 delivery,
  or M12xN16 with exact N4/N8/M tails, explicit `smt.vmadot`, and the selected
  exact E2c3 C8 Q62 `vsmul.e64` requantization/LUT/store path. The
  older E2c route remains available only as a diagnostic control by setting
  `Y26_STAGE52_E2C2=0` before prepare.
- Grouped/depthwise Conv uses prepare-time corrected bias, adjacent-X reuse,
  explicit RVV C8 interior/border handling, and Q62 E2c3 output.
- Input quantization uses explicit RVV and a compact C3 representation consumed
  directly by the RGB stem. The stem accumulates only the 27 real C3
  tap-channel products with tap-major weights.
- Attention uses direct Q/K/V split/transpose addressing, packed static-shape
  integer MatMul, and package-defined exact fixed Softmax with a per-row cache
  for repeated reachable differences.
- LUT, Add, Concat, MaxPool, Resize, Split/reshape/transpose, TopK, and Gather
  have static exact integer implementations.
- The final head has deterministic score-descending, source-index, then class
  tie ordering, uses true N4/N8/N16 output kernels and block-major C8 class
  traversal, and emits `1x300x6`.

The compatibility wake protocol uses condition variables. Setting
`Y26_STAGE53_SPIN_POOL=1` before prepare selects the measured SCHED_OTHER epoch
spin research mode. That mode keeps the same CPU affinity and arithmetic but
occupies the four worker CPUs continuously while the executor is active.
The Stage54 prepared-static, pause, and adaptive-spin candidates were exact but
did not beat raw epoch-spin on complete-model wall time.

The measured call path has no ORT session, Python callback, graph registry,
string dispatch, allocation, file I/O, or float Q/DQ materialization. Input
decode and letterbox are outside pure-model timing and are reported separately.
