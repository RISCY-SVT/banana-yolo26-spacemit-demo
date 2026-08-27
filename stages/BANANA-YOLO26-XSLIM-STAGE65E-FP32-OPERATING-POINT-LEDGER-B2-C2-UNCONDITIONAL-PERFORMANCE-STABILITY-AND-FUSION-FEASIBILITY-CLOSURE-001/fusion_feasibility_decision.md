# Fusion feasibility decision

Status: `not-justified-low-roi-or-unquantified`.

Measured median CPU-tail shares are B2 `0.127700` and C2 `0.128023` of two-stage latency. They establish an upper bound, not a projected exact-tail gain; no tail implementation benchmark is authorized in this Stage.

The best common B2/C2 read-only projected two-stage ORT optimization-level gain versus `ORT_DISABLE_ALL` is `0.000000` at `disable`; the matched two-stage noise floor is `0.014549`. Main-partition timing is unprofiled; the exact accepted per-model tail median is added as an invariant reference only after six-boundary output identity passes. A >=2% option is accepted only when the gain also exceeds that floor and output plus 925-node placement remain exact.

Offline optimization, I/O Binding, EPContext, plugin-tail, and YoloDecode results are capability evidence only. No accepted model, runtime default, or source was changed.
