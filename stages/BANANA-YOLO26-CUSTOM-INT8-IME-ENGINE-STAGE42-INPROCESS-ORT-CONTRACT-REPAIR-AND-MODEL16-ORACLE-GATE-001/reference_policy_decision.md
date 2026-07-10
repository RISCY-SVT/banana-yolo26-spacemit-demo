# Reference Policy Decision

selected_policy: B - fixed host oracle with board integration runtime

## Correctness authority

The authority is host ORT 1.27.0 under the accepted operational session contract (`ORT_ENABLE_ALL`, sequential, 1/1 threads, memory pattern/arena/spinning enabled), with fixed model, cut, input, output, runtime, and manifest hashes.

Board custom scalar and IME are compared directly to saved host-derived integer boundaries. On the primary same-input model4 tensor both are byte-exact. Integer-boundary tolerance is not used or accepted.

## Board ORT role

Board ORT 1.20.2+spacemit is retained for integration, fallback, debug, and timing. It is not the custom-kernel correctness authority because:

- host and board ORT differ under identical `ORT_DISABLE_ALL` inputs at the first QuantizeLinear boundary;
- host and board ORT model4 cuts differ under both DISABLE and ALL;
- host scalar, board scalar, and board IME all agree exactly with the fixed host operational oracle.

Cross-runtime output0 equality remains diagnostic because downstream TopK/Gather/reduction selection is discontinuous. Same-runtime output0 is still a valid regression test under one fixed runtime contract.

## Gate status

- fixed oracle reproducibility: pass.
- board scalar same-input gate: pass, byte exact.
- board IME same-input gate: pass, byte exact.
- board ORT match: fail and explicitly scoped.
- non-zero custom tolerance: not used.
- model16 oracle generation authorization: open; completed in this stage.
