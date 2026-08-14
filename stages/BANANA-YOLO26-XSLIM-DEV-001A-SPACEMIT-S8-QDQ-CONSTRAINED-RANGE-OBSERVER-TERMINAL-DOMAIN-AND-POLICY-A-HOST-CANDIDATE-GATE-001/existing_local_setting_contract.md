# Existing local-setting contract

Audited baseline: `2bc1be073c84ffd8b4e22e372b8f33de4218f9f8`.

Before DEV-001A, `CustomQuantizationParameterSetting` exposed
`input_names`, `output_names`, `max_percentile`, `precision_level`, and
`calibration_type`. `QuantizeConfigRefinePass` traversed operations between
the named boundaries and assigned observer/percentile or precision settings
before `QuantizeFusionPass`, `QuantizeSimplifyPass`,
`ActivationClipRefine`, and block-wise calibration.

The baseline implementation did not provide an exact semantic-tensor
selector, did not emit a final matched-domain manifest, did not fail on
post-fusion root ambiguity, and did not verify that block-wise finetuning or
export retained the intended scale and zero point. Consequently those fields
were useful region hints, but were insufficient evidence that a terminal
YOLO domain had the requested exported qparams.

DEV-001A preserves the old fields. An enabled `range_policy` adds a separate
post-fusion binding and post-calibration finalization path; no enabled policy
means those new passes are absent.
