# Stage 8 Or Stage 7B Next Step Decision

Decision: recommend `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`.

Rationale:

- Stage 7 reached `/model.2/cv1/conv/Conv` and passed host and board correctness.
- Selected-subset IME total is faster than scalar total.
- Activation/requant fallback is now the dominant bucket: `436780 us`, `73.6129%` of IME total.
- Expanding more Conv nodes before reducing activation/requant cost risks hiding the main current bottleneck.

Do not start Stage 8 implicitly. It needs review/approval.
