# Stage64 final report

## Classification

`stage64-xslim211-s8-qdq-vendor-workflow-full-yolo26-int8-ep-correct-measured-and-coco-pass`

Causal conclusion: `vendor-explanation-supported`.

This is vendor-runtime research evidence only. It does not promote XSlim or
SpacemiT ORT, replace the custom executor, alter an accepted release, publish
the generated model, or make a production claim.

## Isolation

Stage64 started from accepted Stage63 commit
`94b86a6bf011cc83fefaf2a960191e97a8daf728` on the dedicated branch
`yolo26-vendor-ort-xslim211-s8-qdq-validation`. Protected main remained at
`1fd2e71bb1d5a924e7c0444cada94f681b73aa91`; tags `v0.9.3-r640` and
`v0.10.0-internal-rd.1` remained at their accepted commits. The custom
executor, accepted releases, `/data/ncnn`, and Stage63 history were read-only.

All generated models, calibration data, COCO predictions, vendor binaries,
Python environments, faults, and raw logs stayed under task-local NVMe
`/data`. Git contains only new validation tooling, compact evidence, and a
small synthetic public repro bundle.

## Tool identities

| Lane | Immutable source | Declared version | Result |
|---|---|---:|---|
| official XSlim | `c246694a1eba8d7689c43ba7b5f469bb0cb29c95` | 2.1.1 | prescribed split works |
| vendor reference | `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` | 2.1.2 | prescribed split works; ReduceMax repaired |

The official PyPI wheel SHA-256 is
`e01cd8b3c7070c038ed60415b30dfe1e35140de0de6725208b2eaa0f871069b3`;
the sdist SHA-256 is
`9804d5c473b9e79f391a645c403fd50dc68f0334ff07b4408692c4359f4f235c`.
The exact vendor-reference wheel SHA-256 is
`eb78f2f1cf98e94b3e214397aaa0bef16fe2ad53d318fe032f990e2f38d6d488`.
Both isolated environments use Python 3.12.3, NumPy 2.4.2, CPU-only
PyTorch 2.10.0, ONNX 1.21.0, host ORT 1.24.3, and ONNXSlim 0.1.87.

An early quantization-summary schema recorded the resolved venv interpreter
target (`/usr/bin/python3.12`) rather than both the venv path and its target.
The hardened rerun records both paths, imports XSlim from the immutable
stage-local venv, and reproduces the selected ONNX byte-for-byte. XSlim is not
installed in the global interpreter. This is an evidence-recorder limitation,
not mixed quantization output.

The board runtime is the unchanged official SpacemiT ORT 2.0.6 archive,
SHA-256
`bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6`.

## Canonical source and calibration

The canonical floating-point model is
`yolo26n_640_e2e_fp32.onnx`, SHA-256
`d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2`.
It is a fixed float32 `1x3x640x640 -> 1x300x6` graph with 453 nodes,
204 initializers, no Q/DQ or QLinear operator, and all six exact
vendor-named split tensors.

The selected calibration list contains 50 deterministic images; its list
SHA-256 is
`4477353183398e4233c2fc46980448b1d3d76c7a533474b1a0673d5917028f1e`.
The 100-image holdout list SHA-256 is
`28eb60b52a5ca8e619f014649cd81c6198b0f11668ea9337efb3adc432e6910d`,
with zero selected-list overlap.
Project-exact preprocessing is byte-identical to the accepted letterbox input
on bus, Zidane, and the canonical image. Vendor-literal preprocessing performs
a materially different direct resize.

The accepted calibration corpus is a 2,015-image subset of COCO val2017.
Consequently, the measured full-COCO result identifies this artifact but is
not an independent calibration-generalization claim.

## XSlim regression boundary

Official XSlim 2.1.1 fails all three bounded two-input/converted ReduceMax
controls with `ValueError: too many values to unpack`. The vendor-reference
commit passes both pure controls exactly and passes Conv-plus-ReduceMax with
maximum absolute error `0.0149146`.

The prescribed six-output YOLO split truncates before the post-processing
ReduceMax. Both lanes therefore complete the split workflow; this is
`bypassed-by-truncation`, not a ReduceMax fix in official 2.1.1.

Direct-E2E diagnostics:

- official 2.1.1: fails during metadata tracing in 16.483 seconds at
  two-input `/model.23/ReduceMax`;
- vendor reference: emits signed-QDQ model
  `f55de815466a32cd6e85e4f18b05e08123d2b8ad92097bcb6a5a4a5dfc9a95b7`,
  but all 100 host holdout outputs have score channels collapsed to zero.

The vendor direct artifact is rejected before board execution or COCO. This
supports the prescribed branch-separation and unquantized-tail policy.

## Quantization and graph contract

The selected official project-exact model SHA-256 is
`29e08be834afb8925ca02af69d9a25df05449e9367ef3d8dd8ca4d57cf59a4fb`.
Its split inference SHA-256 is
`ac855266d6ecaf092748692c10dd26548d0fa6c449d48bb7cc2b988257412d6c`;
the floating-point tail SHA-256 is
`18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3`.

Measured conformance:

| Property | Value |
|---|---:|
| Q/DQ nodes | 812 |
| QLinear operators | 0 |
| UINT8 zero points | 0 |
| signed activation sites | 710 |
| signed symmetric per-channel weight sites | 102 |
| Conv with explicit valid `kernel_shape` | 102 / 102 |
| split inference nodes | 1,161 |
| floating-point CPU-tail nodes | 34 |

All weight zero points are signed INT8 zero. Activation zero points are signed
INT8 scalars and include zero and nonzero values. Bbox and confidence/class
branches remain separate across the six vendor boundaries. The post-processing
tail is unquantized and explicitly CPU-owned.

The official and vendor-reference lanes produce byte-identical models for
both project-exact and vendor-literal split configurations.
The selected official model also reproduces byte-for-byte on a second run.
XSlim's generated analysis Markdown differs between those runs, so the whole
output tree is not byte-identical even though the deployable ONNX is.

## Host correctness

The unsplit FP32 model and FP32 split-plus-tail pipeline reproduce exactly.
All four mandatory split configurations pass ONNX checking, host execution,
finite/range checks, and the score-collapse gate. The selected 100-image
holdout has no all-zero confidence/class branch; final-output cosine
similarity to same-source FP32 is `0.859751106535`.

## Board support contract

Independent synthetic controls establish the representation boundary:

| Control | SpacemiT ORT 2.0.6 result |
|---|---|
| S8 QDQ Conv, zero activation zero point | assigned, exact |
| S8 QDQ Conv, nonzero signed activation zero point | assigned, exact |
| S8 QDQ Conv, per-tensor weights | assigned, exact |
| S8 QDQ Conv without `kernel_shape` | exact CPU fallback; placement not proven |
| S8 QDQ MatMul, zero/nonzero signed zero point | assigned, exact |
| U8 QDQ Conv and MatMul | unsupported error followed by SIGABRT |
| historical QLinear Conv and MatMul | accepted Stage63 SIGILL negatives |

All supported S8 controls pass affinity gates on CPU0, CPU0-3, CPU4, CPU4-7,
and CPU0-7. One bounded GDB packet records the UINT8 abort in libc
`__pthread_kill_implementation`; no new SIGILL class was found.

The unchanged Stage63 plugin surface passes one official-sample load smoke and
ten exact independent-plugin dispatches. The accepted 1,000-run plugin result
is reused by exact runtime-tree identity.

## Full YOLO26 placement

The selected S8 inference session creates on both CPU EP and SpacemiT EP. ORT
profiling exposes the complete 1,161-node inference graph as one
`SpaceMITExecutionProvider` Spine subgraph, with no observed CPU provider
event in that inference session. The source graph contains 102 Conv and four
MatMul nodes. The separate 34-node float tail runs on CPU by design and is not
counted as fallback.

The provider profile does not expose a source-node duration map inside the
closed fused subgraph. Stage64 therefore proves meaningful whole-inference
placement, not individual source-node timing or internal implementation.

F0, bus, and Zidane complete on CPU and SpacemiT surfaces with finite six
boundaries and finite `1x300x6` output. CPU and EP tensors are compared
numerically; byte identity is neither assumed nor claimed.

## COCO

All decision surfaces complete with zero image failures and zero non-finite
outputs:

| Surface | Images | mAP50-95 | Prediction SHA-256 |
|---|---:|---:|---|
| same-source FP32 CPU | 5,000 | 0.404730651123 | `e8c97ebf...96c1a` |
| XSlim 2.1.1 project-exact S8 CPU split | 5,000 | 0.358768508793 | `6162fc26...3a00` |
| XSlim 2.1.1 project-exact S8 SpacemiT split | 5,000 | 0.359072683728 | `8d16e1cd...94a8` |

The selected EP route records mAP50 `0.514173727408`, AP-small
`0.179166788668`, AP-medium `0.420416372236`, and AP-large
`0.516591419978`. Its mAP50-95 loss against same-source FP32 is 0.045657967395,
or 4.5658 AP points. Official and vendor-reference S8 evidence is reused only
where model and runtime identities are byte-equal.

## Performance and stability

The fixed-input 500-inference placement arm enabled ORT profiling. Its selected
SpacemiT means are:

| Component | Mean us | p95 us | p99 us |
|---|---:|---:|---:|
| S8 inference | 109,357.871 | 115,596.896 | 144,298.043 |
| CPU tail | 14,855.441 | 15,698.535 | 18,307.698 |
| two-stage total | 124,213.313 | 130,702.184 | 162,601.658 |

With the same profiling enabled, the matched CPU control records
3,216,303.360 us inference, 17,419.199 us tail, and 3,233,722.559 us total
means; total p95 is 3,309,925.630 us and p99 is 3,399,719.179 us.

Primary CPU-versus-EP steady attribution uses the no-profile 5,000-image COCO
timing streams over the same image list. CPU means are 3,021.905 ms
inference, 17.607 ms tail, and 3,085.385 ms total. SpacemiT means are
99.493 ms inference, 14.986 ms tail, and 158.778 ms total. This is a 30.37x
inference-subgraph speedup and a 19.43x complete two-stage speedup. The
fixed-input profile arm is retained for placement evidence and is not
silently presented as an uninstrumented runtime surface.

A separate profiled CPU stability process completed 603 outputs with one
stable fixed-input hash, then stopped making forward progress while consuming
about 1.7 GiB RSS and sustained CPU. It was terminated and retained as
`cpu-stability1000-profiled-partial`; it is not selected evidence. The
unprofiled CPU COCO process completed 5,000 consecutive varied-image
inferences with zero image failures and zero non-finite outputs, exceeding
the required 1,000-run CPU stability count without profiler distortion.

The selected SpacemiT 1,000-run stability arm has a 121,689.438 us total mean,
124,997.956 us p95, 130,516.512 us p99, and one stable output hash. The
unprofiled 10,000-run soak has a 121,765.490 us total mean, 125,058.193 us p95,
130,894.930 us p99, 138,394.950 us p99.9, 164,434.795 us maximum, and stable
hash `0x198e555969cee70e`.

The full EP COCO pipeline averages 0.1066 ms decode, 44.1929 ms preprocessing,
99.4927 ms inference, 14.9856 ms tail, and 158.7777 ms total. These are
preloaded image-pipeline measurements, not camera FPS or sensor latency.

The provider creates fixed CPU0/1/2/3 workers while the controller remains
allowed on CPU4-7. All sampled CPU frequencies are 1.6 GHz. No board thermal
zone is exposed, so temperature is unavailable rather than inferred.

## Causal decision

`vendor-explanation-supported`

The historical Stage63 U8-QDQ model remains a valid unsupported-format
negative control. A newly generated, vendor-conforming S8-QDQ split model
passes host semantics, tiny exact controls, complete SpacemiT inference
placement, fixed fixtures, stability, a 10,000-run soak, and full COCO. This
supports quantization representation as the cause of the prior full-model
compile failure.

It does not resolve two separate defects:

- unsupported U8 inputs still terminate the process instead of returning a
  contained status;
- official XSlim 2.1.1 still has the tested two-input ReduceMax parser defect.

The measured 4.5658-point accuracy loss and calibration/evaluation overlap
also require human review before any promotion study.

## Closure

The issue update is a draft only. No GitHub issue was modified. Protected
refs, accepted releases, the custom executor, and system state are unchanged.
The public repro bundle contains only synthetic tiny controls, independent
oracles, a neutral runner, and sanitized result tables.

Decision: **the XSlim S8-QDQ plus ORT 2.0.6 route deserves a separate human
promotion review, but Stage64 performs no promotion.**
