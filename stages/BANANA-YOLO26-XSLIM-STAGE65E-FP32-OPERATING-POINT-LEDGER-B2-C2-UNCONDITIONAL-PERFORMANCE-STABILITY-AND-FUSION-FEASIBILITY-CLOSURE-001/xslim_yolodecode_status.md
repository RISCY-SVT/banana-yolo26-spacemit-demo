# XSlim YoloDecode status

Status: `source-present-frozen-split-does-not-use-it`.

Current XSlim source contains the YoloDecode fusion implementation: `yes`. The frozen B2/C2 inference graphs intentionally stop at six head boundaries and contain no `YoloDecode` node.

Stage64's exact `direct_e2e_diagnostic.tsv` (SHA-256 `064b9b89086f0a9c3bdcc84420a9f084f3c9a72fc13c4e3e8e0e9f4765818422`) binds the repaired vendor-reference direct-E2E lane to 100 holdout images, zero passes, and 100 score collapses. The exact causal decision (SHA-256 `15c892bc2529900dcc91a591187110943ce4483fbb3e85b04e5388abc0ab1565`) binds those cases to finite `1x300x6` outputs with zero nonzero scores. The defect remains unreconciled, so direct-E2E generation is not justified by this audit.

A future exact candidate would have to preserve the current source/model/qparams, reproduce the 34-node tail exactly, retain finite noncollapsed task output, and re-prove provider placement and COCO accuracy.
