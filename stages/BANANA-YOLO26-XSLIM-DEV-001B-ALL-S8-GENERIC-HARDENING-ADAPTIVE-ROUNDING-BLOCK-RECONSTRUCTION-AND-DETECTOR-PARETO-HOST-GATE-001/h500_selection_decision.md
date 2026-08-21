# H500 Selection Decision

The shared 10,000-draw paired bootstrap used seed `65006` for all candidate
comparisons. The draw matrix and replicate payload identities are preserved in
the raw stage root. No candidate passed every predeclared H500 qualification
and A1 Pareto condition.

## Candidate decisions

| Candidate | mAP delta vs B2 | P(delta > 0) | Decisive gate result |
|---|---:|---:|---|
| C2_T6_RANK_QP | +0.010772006184 | 1.0000 | Fail: `P(AR-small delta >= -0.005)=0.8361`, below 0.90 |
| C3_R7_BR | +0.005907517921 | 0.9915 | Fail: `P(AP-medium delta >= -0.005)=0.8722`; mAP is 0.001155428169 below A1, beyond the 0.001 Pareto allowance |
| C4_R0_BR | +0.008477727558 | 0.9884 | Fail: AP-small and AP-medium point deltas are -0.009175642319 and -0.009918593514 |
| C5_COMBINED | +0.003878666218 | 0.8936 | Fail: mAP point delta is below +0.004 and probability is below 0.95 |

C2 provides the strongest bounded H500 accuracy signal and improves AP-large
by 0.022827083953, but it does not satisfy the recall non-inferiority contract.
C3 improves AR-large but misses both an AP-medium probability gate and the A1
mAP Pareto allowance. These observations are retained as diagnostics, not as a
candidate qualification.

## Disposition

The full-val2017 gate remains closed. No C6 or later observer/reconstruction
sweep is authorized. The generic hardening commits are retained, while this
all-S8 PTQ reconstruction lane is closed for YOLO26.
