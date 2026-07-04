# Bottleneck Decision

classification_basis: `stage10-backbone-expanded-ready-for-branch-stage`

## Observed Stage 10 A2 Timing

- total: `234341 us`
- activation total: `36039.4 us`
- activation share: `15.379%`
- conv share: `83.9864%`
- pack/layout share: `0.482372%`
- split + branch share: `16.7566%`

## Dominant Bucket

The dominant bucket after Stage 9 is `conv / IME`, not activation/requant, Split copy, or pack/layout.

## Reasoning

Stage 10 added Conv2 activation, Split output 1 handoff, and `/model.2/m.0/cv1/conv/Conv`. Correctness passed on host and CPU0-3 board runs. The expanded subset remains substantially faster than scalar reference for the same selected contract, and activation remains below the 20% hypothesis threshold on the old subset and below 40% on the expanded subset.

## Next Stage Direction

Proceed to a bounded branch stage that either:

- adds `/model.2/m.0/cv1` activation and `/model.2/m.0/cv2/conv/Conv`, or
- integrates the residual Add and Concat only after branch Conv activation contracts are oracle-checked.
