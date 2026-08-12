# Stage65B-R2 final report

Classification: `stage65b-r2-independent-selection-pass-upstream-branch-error-material-early-subgraph-r3-ready`

Publication classification: `research-branch-evidence-only-no-board-claim`

## Recovery and identity

The host-side work resumed from verified immutable R1 evidence. No accepted
model was regenerated. The direct user clarification attributes the historical
reboot to the external Windows 10 host; the incomplete R1 B3 tree remained
isolated and its clean rerun had already reproduced byte-for-byte.
All resumed PTQ, evaluator, and bootstrap processes exited cleanly; no stage
worker remains active at closure.

## Independent H500 selection

- Winner: `B2` at `0.4446654879525213` mAP50-95.
- Runner-up: `B0` at `0.4375988415074177`.
- Point delta: `0.00706664644510363`.
- Winner/runner bootstrap 95% interval:
  `-0.0013079212367526015` to
  `0.016270033970465735`; P(delta>0)
  `0.96`.
- Scout/full-val metrics were not used for selection.

## FP32 reconciliation

`imported-fp32-surface-confounded`. F0, F1, and H8 are byte-identical in the current runner on
H500 and full val2017. The older imported FP32 prediction surface is therefore
confounded by historical harness/serialization behavior, not a proven split
residual.

## D8 causal diagnostic

- Classification: `upstream-branch-error-material`.
- H500 D8 mAP50-95: `0.45621089142110244`.
- Full-val D8 mAP50-95: `0.3793441320923446`.
- Recovery fractions: H500 `0.335397387`, full val
  `0.374970551`.

Bypassing only the final six output Q/DQ pairs recovers a minority of the gap;
earlier branch quantization error is material. D8 is diagnostic-only.

## B2 robustness and route

Variance decision: `no-significant-aggregate-map-sensitivity-proven`. Selected later route: `R3-early-subgraph-localization-ready`.
Vseed crossed the +0.005 H500 point gate but missed the predeclared
P(delta>0)>=0.95 full-val gate (observed 0.94). Vdraw did not change aggregate
mAP significantly, although its AP-small/AP-medium bootstrap intervals show a
membership-dependent size-bin signal.
No targeted model was generated and no board, provider, performance, soak, or
promotion claim is made.
