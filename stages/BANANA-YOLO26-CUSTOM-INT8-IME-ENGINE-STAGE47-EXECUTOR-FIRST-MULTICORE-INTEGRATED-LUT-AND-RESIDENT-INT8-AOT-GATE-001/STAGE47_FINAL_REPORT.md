
# Stage47 final report

classification: stage47-blocked-correctness
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE47-EXECUTOR-FIRST-MULTICORE-INTEGRATED-LUT-AND-RESIDENT-INT8-AOT-GATE-001
start_head: da213f6c11339187d2169e5a2516feef1b732dd9
end_head: pending-local-commit-see-final-response

## Proven

- Exact graph census: 106 compute nodes, 102 Conv, 4 MatMul/Gemm, 2,740,153,600 MACs.
- M4/M8/M12 execute complete full shapes and exact tails on nine deterministic graph cases.
- M12 CPU0-3 model5 scales to `8.940790 GMAC/s` with `90.896728%` efficiency.
- Resident model4-model8 schedule: 29 operations, 1,638,400-byte arena, 880,128 packed-weight bytes, zero ORT/internal transpose/float QDQ in the measured run.
- Host custom scalar and board IME are byte-identical; CPU4-7 execute no IME and no SIGILL occurred.
- RNE/RTZ/RDN/RUP/RMM produce one stable hash and restore the original FRM.
- B120 ORT resource-matched slice is `61239.054678 us`; custom with adapters is `135761.369456 us`.

## Broken

- F0-F3 and F6-F7 eventually diverge from fixed-host ORT; F4-F5 remain exact through model8.
- Focused F2 divergence is a dequantized-float Conv tie surface, not scalar/IME or pair-saturation error.
- The custom slice is `121.690831%` slower than B120 ORT with adapters.

## Unknown

- A production integer-semantic export contract is not yet fixed.
- Student 416/512 accuracy and measured latency remain unknown; neither is selected.

## Decision

The mandatory no-tolerance fixed-host gate fails, so the classification is
`stage47-blocked-correctness`. Independently, `97.268007%` MAC mapping gives
`225.228/1168.165/1277.559 ms`
optimistic/central/conservative analytical estimates; current 640 is not target
credible on this substrate. Proceed to one semantic-contract and student
architecture-preparation gate. This is not a full engine, production result, or FPS claim.
