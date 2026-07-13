# K1X INT8 Executor Architecture

The executor is a static AOT runtime for one frozen YOLO26n-640 profile.

- One global 64-byte-aligned activation arena uses lifetime-based offsets.
- Feature tensors use `NCHWc8_SPATIAL_INNER_V1`; attention/head tensors use a
  fixed row-major integer layout where required.
- Immutable packed weights and arithmetic assets are loaded during prepare.
- Four persistent workers are pinned to CPU0-3. CPU4 is controller-only.
- Dense Conv uses M12xN16 with exact tails, direct C8 A delivery, `smt.vmadot`,
  and exact E2c Q62 `vsmul.e64` requantization.
- Grouped/depthwise Conv uses a direct NCHWc8 integer path.
- Attention uses packed static-shape integer MatMul and package-defined fixed
  Softmax.
- LUT, Add, Concat, MaxPool, Resize, Split/reshape/transpose, TopK, and Gather
  have static exact integer implementations.
- The final head has deterministic score-descending, source-index, then class
  tie ordering and emits `1x300x6`.

The measured call path has no ORT session, Python callback, graph registry,
string dispatch, allocation, file I/O, or float Q/DQ materialization. Input
decode and letterbox are outside pure-model timing and are reported separately.
