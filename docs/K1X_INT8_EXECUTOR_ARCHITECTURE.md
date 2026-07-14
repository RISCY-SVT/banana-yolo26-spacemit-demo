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
- Dense Conv uses M12xN16 with exact tails, direct C8 A delivery, `smt.vmadot`,
  and the selected exact E2c2 Q62 `vsmul.e64` requantization/store path. The
  older E2c route remains available only as a diagnostic control by setting
  `Y26_STAGE52_E2C2=0` before prepare.
- Grouped/depthwise Conv uses an explicit RVV C8 interior kernel plus exact
  bounded border handling and Q62 E2c2 output.
- Input quantization writes NCHWc8 directly with explicit RVV. The RGB stem
  accumulates only the 27 real C3 tap-channel products with tap-major weights.
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

The measured call path has no ORT session, Python callback, graph registry,
string dispatch, allocation, file I/O, or float Q/DQ materialization. Input
decode and letterbox are outside pure-model timing and are reported separately.
