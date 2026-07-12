# Stage 49 final report

classification: `stage49-persistent-nchwc8-slice-net-positive`

## Proven

- Start HEAD and clean workspace matched `b163c83f7dc1677c8b31b9a2cc75e227d5992b0d`.
- `K1X_INT8_V1` package integrity, external manifest trust, loaded-byte sum/bound recomputation, little-endian enforcement, and alias rejection pass.
- Stage48 model5 regression passes: 6458.369394 us mean, 7352.872650 us p95.
- F0-F7 Python/C++/board-scalar/board-IME equality is exact at all 17 slice boundaries; FRM restores; no SIGILL; IME is CPU0-3 only.
- T6 + M12xN16 + P3 + E1 + spatial partition + four workers is selected. Model5 is 5145.442776 us mean, 7663.699800 us p95, 45.852147 GMAC/s.
- Persistent model4-final through model6 internal slice is 26710.414338 us mean and 26845.251100 us p95 versus B120 ORT 42036.659040/42078.047838 us. Mean delta is -36.459236% with zero internal conversions and zero float materializations.

## Broken

- Exact RVV Q62 E2 is not available: GCC leaves the 64x64-to-128 loop scalar and no proven RVV sequence was accepted.
- Scalar entry/exit converters add 38040.223614 us; the with-adapters diagnostic surface is slower than ORT.
- Generic cache PMU events and X60 named stall/vector events are unavailable on the current board tool/source surface.

## Unknown

- Full model resident-executor latency and accuracy are unmeasured.
- Stable LUT-v2 rows for RGB stem, N4/N8 outputs, and wider model7/8 composition remain open.
- CPU-wide PMU command-envelope counts do not isolate kernel cycles or per-worker IPC.

## Correctness policy

`K1X_INT8_V1` remains the deployment authority. Legacy host float-QDQ and B120 ORT are diagnostic only.

## Timing status

Headline timing excludes package prepare, validation, file I/O, ORT, adapters, and phase instrumentation. Adapter and phase diagnostics are reported separately. The internal slice passes the predeclared net-positive gate; this is not a full-model or full-frame result.

## Implementation authorization

The route remains experimental/non-default. No production dispatch, full graph executor, student design/training, RT205 work, CPU4-7 IME, or push is authorized or claimed.

## Next readiness

Proceed to a bounded persistent NCHWc8 model4-final-to-model8 LUT-v2 closure stage. Keep both student hypotheses held.
