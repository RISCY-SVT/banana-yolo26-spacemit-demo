# Multi-Shape Threading Sidecar Report

Primary integrated target:

```text
/model.4/m.0/cv1/conv/Conv
1x80x80x32 -> 1x80x80x16
3x3 stride1 padding1
status: integrated sidecar
```

Additional shapes:

```text
/model.4/cv1/conv/Conv: not_attempted_by_scope
/model.4/m.0/cv2/conv/Conv: not_attempted_by_scope
```

Reason:

Stage18 is an implementation sidecar for the Stage17 proven representative/full-shape target. Additional full-shape oracles for the 1x1 and second 3x3 Conv were not required for the primary acceptance gate, and Stage18 did not expand graph coverage.
