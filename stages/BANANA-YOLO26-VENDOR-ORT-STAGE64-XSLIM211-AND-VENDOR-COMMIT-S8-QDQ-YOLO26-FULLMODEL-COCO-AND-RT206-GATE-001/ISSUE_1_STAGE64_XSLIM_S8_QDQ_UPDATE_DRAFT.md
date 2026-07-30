# Draft update for spacemit-com/onnxruntime issue #1

Do not post automatically. This is a sanitized draft for human review.

## Environment

- Banana-Pi BPI-F3 / SpacemiT K1X, Bianbu 2.2.1, Linux 6.6.63
- official ORT asset: `spacemit-ort.riscv64.2.0.6.tar.gz`
- asset SHA-256:
  `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6`
- official XSlim 2.1.1 commit:
  `c246694a1eba8d7689c43ba7b5f469bb0cb29c95`
- vendor-referenced XSlim commit:
  `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`

## Result

The historical U8-QDQ/QLinear failures remain reproducible as unsupported
negative controls, but a newly generated vendor-conforming signed-INT8 QDQ
YOLO26 split works:

| Surface | Result |
|---|---|
| S8 QDQ Conv, explicit `kernel_shape`, signed zero/nonzero activation ZP | SpacemiT assigned; exact |
| S8 QDQ MatMul, signed zero/nonzero activation ZP | SpacemiT assigned; exact |
| S8 QDQ Conv without `kernel_shape` | output exact, but CPU fallback observed |
| U8 QDQ Conv/MatMul | unsupported provider error followed by SIGABRT |
| official XSlim 2.1.1 two-input ReduceMax | still fails |
| XSlim vendor commit two-input ReduceMax | fixed in bounded controls |
| official 2.1.1 prescribed six-output split | works |
| complete S8 inference graph | one profiled SpacemiT subgraph |
| explicit float post-processing tail | CPU by design |

The selected generated graph has 812 Q/DQ nodes, zero QLinear operators, zero
UINT8 zero points, signed per-tensor activations, 102 signed symmetric
per-channel weight sites, and explicit valid `kernel_shape` on all 102 Conv
nodes.

Fixed fixtures, 1,000-run stability, and a 10,000-run soak pass. The full
SpacemiT pipeline completes COCO val2017 5,000/5,000 with mAP50-95
`0.35907268372810625` and prediction SHA-256
`8d16e1cdc0436c0ff8f5dae0d411778bca70c870206db55b6ef25d4c7af494a8`.
The same-source FP32 control is `0.40473065112282053`.

The matched 500-run SpacemiT two-stage mean is 124,213.313 us
(109,357.871 us inference plus 14,855.441 us CPU tail). The 10,000-run total
mean is 121,765.490 us with stable output hash.

The independent CPU S8 full-COCO result is mAP50-95
`0.35876850879267863`, prediction SHA-256
`6162fc26a654f19e21a7ba65f064ab1c3f651a318453944e25026f2e75ae3a00`,
with zero image failures. EP mAP50-95 is 0.000304174935 higher; prediction
counts and hashes differ, so task-level agreement is claimed rather than
byte identity.
Direct-E2E diagnostics are: official 2.1.1 fails metadata tracing at
two-input `/model.23/ReduceMax`; vendor-reference
generates a signed-QDQ graph but collapses every score channel to zero on all
100 host holdout images. That diagnostic is rejected before board execution.

## Interpretation

This evidence supports the vendor explanation that the old full-model failure
was representation-specific: the old model was U8-QDQ, while the supported
S8-QDQ split executes. Stage63 remains valid for its supplied model and
operators.

Two issues remain:

1. Unsupported U8 inputs terminate the process after a provider error instead
   of returning a contained status.
2. Official XSlim 2.1.1 still fails the tested two-input ReduceMax path; the
   prescribed split bypasses that path, while commit `9a33f2f` fixes it.

The measured S8 model loses about 4.57 AP points against same-source FP32.
Its calibration corpus overlaps COCO val2017, so this is artifact
characterization rather than independent generalization evidence.

## Attachments

`ISSUE_1_STAGE64_MINIMAL_S8_QDQ_REPRO_BUNDLE.tar.gz` contains only synthetic
tiny models, independent inputs/oracles, neutral runner source, sanitized
tables, and checksums. It contains no full YOLO model, weights, calibration or
COCO data, custom executor source, vendor binary, credential, or private path.

Bundle SHA-256:
`c73f1807dfa2edfd3b82b2524fecbda004c9cac86af3824062d2498df16ab47d`.

Questions for the vendor:

1. Can unsupported UINT8 graphs return a catchable status instead of aborting?
2. Which public XSlim release will include the two-input ReduceMax repair?
3. Is one fused SpacemiT inference subgraph the expected profiling surface for
   this workflow, and is a source-node assignment map available?
