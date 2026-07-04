# Pre-registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-RVV-REQUANT-LUT-FUSION-GATE-001`
created_before_coding: true

## H1

The Stage 8 residual activation/requant cost is dominated by scalar per-element int32 -> uint8 conv-output-code quantization and scalar table lookup/write, not by SiLU math.

## H2

Removing avoidable per-element scalar overhead in conv-output-code quantization should reduce activation/requant time before any packA fusion.

## H3

RVV or scalar-unrolled requant+lookup may reduce activation/requant time; however vrgather/vluxei performance on X60 is unknown and must be measured rather than assumed.

## H4

Fusing requant -> LUT -> packA is useful only after the remaining scalar requant/lookup path is characterized or improved. Fusion is a secondary Stage 9 candidate, not the first implementation assumption.

## H5

Exact Stage 8 activation semantics must be preserved for accepted paths: mismatches=0 against Stage 8 scalar reference and Stage 7 fixture oracle. Approximation candidates are sidecar only and cannot be selected without explicit tolerance report.
