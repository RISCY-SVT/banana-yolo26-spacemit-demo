# XSLIM-DEV-001B Final Report

## Classification

```text
xslim-dev-001b-all-s8-reconstruction-no-pareto-candidate-
close-this-ptq-lane
```

Publication classification: `not-authorized-not-attempted`.

Stage ID: `BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001`.

## Input Closure

- Stage65C-R1 packet: `8398831b147cc890436e968d830b14c0d5347ee5a24946b03156c66aa08b22e6`, 63 files, 1,983,169 bytes.
- B2 deployable/inference: `0e7040d4...` / `40ba6a7f...`.
- A1 deployable/inference: `8fad9fa0...` / `f7c5345f...`.
- Common tail: `18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3`.
- Deterministic C200 partition: C160 `a7588e76...`, C40 `a76237de...`; overlap with H500 and val2017 is zero.
- Append-only Stage65C-R1 errata were added without changing the historical classification. No distinct current-stage host reboot occurred; the historical defect was board awk portability and zero-byte thermal files remain non-evidence.

## Generic XSlim Result

XSlim advanced from `3e275c6496d603d3f75f363ed00aa633ffc00408`
to `46d5d36bcb6979bab6567fb4fe62839689f1881c`, tree
`1788779cd0887a1c8e6924cd63ad7d16d42f41ca`, version
`2.1.2+riscy.2.dev2`. GitHub and GitLab match that head.

The implementation passes the model-independent contracts for legacy versus
constrained settings, `lock_qparams=true/false`, complete final constraints,
multiplicity-preserving small-array KL, stratified activation sampling,
SpacemiT structural profile hardening, deterministic block training, adaptive
rounding, held-out early stopping/rollback and topology-preserving bias
correction. Activation drop is implemented behind a default of `0.0`; the
bounded reconstruction selection retained `0.0`.

No-override generation reproduces frozen B2 byte-for-byte for deployable and
inference ONNX and preserves the exact common tail.

## Tests And Package

- Full pytest: 207 passed, 4 inherited warnings, 65 subtests.
- Focused reconstruction: 16 passed.
- Banana tooling: 12 passed.
- Ruff, `compileall`, focused strict mypy, `pip check`, CLI and fresh install
  smokes pass.
- Wheel: `7b2ca5075b90643a89da8c0529b20faf0b87c8cace8a2fddc6a4b01ae421c6d8`, two clean byte-identical builds.
- Raw sdist hashes: `9228f9f9...` and `7b2082df...`; normalized extracted identity `a5bf928823cf459486b611e8d8abea00586fa1903b8d1dd9337f73c6e8d5a9f1`.
- No tag, release or package publication occurred.

## Candidate Matrix

| Lane | Deployable SHA-256 | Inference SHA-256 | H500 mAP | AP-large | AR-large |
|---|---|---|---:|---:|---:|
| B2 | `0e7040d4...` | `40ba6a7f...` | 0.444665488 | 0.676800104 | 0.829328215 |
| A1 | `8fad9fa0...` | `f7c5345f...` | 0.451728434 | 0.688792479 | 0.824114772 |
| C2_T6_RANK_QP | `e963be11...` | `281f4acd...` | 0.455437494 | 0.699627188 | 0.828629474 |
| C3_R7_BR | `39daacc4...` | `cb0bd957...` | 0.450573006 | 0.698960707 | 0.837690916 |
| C4_R0_BR | `0e32a39f...` | `4e15a7b...` | 0.453143216 | 0.699888859 | 0.844793368 |
| C5_COMBINED | `acaf8b63...` | `875c7b0e...` | 0.448544154 | 0.700349096 | 0.848459985 |

All four generated candidates are exact across two clean generations and pass
the same 812-QDQ topology, 0 QLinear, 0 UINT8, 0 FP16, 102/102 Conv
`kernel_shape`, six-output and exact-tail gates. No new FP island or QDQ
boundary exists.

## H500 Decision

The shared 10,000-replicate bootstrap uses seed `65006`; draw-matrix SHA-256 is
`932375fad24bca092958d3ce076da826eca023491b29227a18b811b524cdb0cf` and
replicate NPZ SHA-256 is
`f00385b17a2e17105b6ac95bc60912bbee93812e800453c0139c70a68765d4d4`.

C2 improves mAP over B2 by `+0.010772006184` with `P(delta>0)=1.0`, but
`P(AR-small delta >= -0.005)=0.8361`, below the required 0.90. C3 fails the
AP-medium non-inferiority probability (`0.8722`) and A1 Pareto mAP allowance.
C4 fails AP-small/AP-medium point limits. C5 misses the mAP point and
probability thresholds.

No candidate qualifies; full val2017 was not opened and no full-val process or
prediction was created. Per the stop rule, no later PTQ candidate or observer
sweep was started.

## Protected State And Disposition

Banana protected main, custom executor tree and `/data/ncnn` head/tree/diff and
three pre-existing dirty paths are unchanged. XSlim upstream main and the
published release tag are unchanged. No board or custom-executor command ran.

The generic hardening is retained for review. No YOLO26 candidate from this
matrix is ready for a K1X gate. Further accuracy work requires separately
authorized head-only QAT or model co-design.

Timestamp: `2026-08-21T00:59:58Z`.
