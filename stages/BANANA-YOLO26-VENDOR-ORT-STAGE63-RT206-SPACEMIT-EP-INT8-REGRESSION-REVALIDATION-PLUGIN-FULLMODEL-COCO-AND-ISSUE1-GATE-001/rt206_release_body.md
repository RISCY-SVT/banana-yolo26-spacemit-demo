 ## Features
  - Added BatchGemm graph fusion and operator test coverage.
  - Added support for `WindowPartition` and `WindowReverse`.
  - Added the `layernorm nc` kernel path.
  - Added direct A100 `Conv2D k5s1` support.
  - Improved threading performance for `Slice` copy mode.

  ## Fixes
  - Fixed dynamic quantized dense output dtype handling.
  - Fixed the default `alpha` behavior for `HardSigmoid` / `HardSwish`.
  - Fixed transpose issues when `axis dim = 1`.
  - Fixed memory leak and memory pool corruption issues.

  ## CI / Tooling
  - Stabilized the release plugin packaging workflow.
  - Added support for configurable perf test cases.

  ## Tests
  - Added and updated unit tests for Gemm, BatchGemm, LayerNorm, Window ops, Resize, Slice, Unary ops, and transpose-related cases.
  - Expanded benchmark coverage for Gemm and MultiHeadMatMul.
