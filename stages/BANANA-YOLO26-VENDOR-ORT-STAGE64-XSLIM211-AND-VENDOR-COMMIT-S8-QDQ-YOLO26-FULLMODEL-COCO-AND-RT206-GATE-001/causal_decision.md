# Causal decision

## Result

`vendor-explanation-supported`

The historical Stage63 full-model failure does not reproduce when the same
YOLO26 lineage is quantized from its floating-point source into the
vendor-described signed INT8 QDQ representation and split before
post-processing. The selected official XSlim 2.1.1 project-exact graph:

- contains 812 Q/DQ nodes and zero QLinear operators;
- contains no UINT8 zero point;
- uses per-tensor signed activation quantization;
- uses 102 symmetric signed per-channel weight sites with zero point 0;
- has explicit `kernel_shape` on all 102 Conv nodes;
- retains the six separate vendor-named bbox and confidence/class boundaries;
- executes its complete inference session as one profiled SpacemiT subgraph;
- executes the floating-point post-processing tail on CPU by design;
- passes host semantics, fixed board fixtures, a 1,000-run stability arm, a
  10,000-run soak, and COCO val2017 5,000/5,000.

This supports a representation-specific cause for the old U8-QDQ full-model
failure. It does not reclassify Stage63 as erroneous: Stage63 accurately
measured the supplied U8 graph and historical QLinear controls.

## XSlim release boundary

Official XSlim 2.1.1 still fails two-input `ReduceMax`; the vendor-referenced
commit `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` fixes the three bounded
ReduceMax controls. The prescribed six-output split truncates before that
post-processing operator, so the defect is bypassed rather than fixed in the
official release.

For both vendor-literal and project-exact preprocessing, the official and
vendor-reference lanes emit byte-identical final models. The later commit is
therefore not required for the prescribed split, although it is required for
the tested direct `ReduceMax` path.

## Limits

The selected S8 route reaches mAP50-95 `0.35907268372810625`, versus
`0.40473065112282053` for the same-source FP32 control, a loss of about 4.57
AP points. The 50-image calibration list is disjoint from the 100-image
holdout but is drawn from a corpus that overlaps COCO val2017. The COCO result
is valid for this generated artifact but is not an independent calibration
generalization claim.

Unsupported U8 controls still terminate with SIGABRT after provider errors,
and accepted QLinear controls still terminate with SIGILL. A supported path
does not remove those failure-containment defects.

No runtime promotion, release change, external model publication, or
production claim follows from Stage64.
