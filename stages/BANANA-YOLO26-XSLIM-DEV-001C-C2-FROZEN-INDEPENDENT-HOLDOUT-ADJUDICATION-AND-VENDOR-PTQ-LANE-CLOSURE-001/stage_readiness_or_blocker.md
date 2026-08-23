# Stage readiness and blocker

The fresh H5000 gate passed and conditional full val2017 was completed. C2 passed the complete universal accuracy and size-bin non-inferiority contract against B2.

C2 is not ready for a K1X gate because the frozen Pareto repair contract against A1 failed on full val2017:

- C2-A1 AR-large point: `+0.001120178248`, required `>= +0.002`.
- `P(C2-A1 AR-large > 0)`: `0.876`, required `>= 0.95`.

The vendor all-S8 PTQ lane is closed. No inconclusive result automatically opens another PTQ Stage.

Deferred routes requiring separate authorization are:

1. head-only S8 QAT;
2. model/executor co-design;
3. custom-executor rank-aware terminal calibration;
4. stable XSlim source-hardening closure.
