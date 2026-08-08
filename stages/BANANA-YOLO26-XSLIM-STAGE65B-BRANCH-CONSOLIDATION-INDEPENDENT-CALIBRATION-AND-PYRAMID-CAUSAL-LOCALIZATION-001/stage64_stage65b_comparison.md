# Stage64 to Stage65B comparison

Stage64 remains the accepted supported-path runtime study, but its selected
50-image calibration list came from a 2,015-image COCO val2017 subset. Its
host S8 CPU result was mAP50-95 `0.35876850879267863`; same-source FP32 was
`0.40473065112282053`.

Stage65B was intended to separate corpus effects from policy effects. No
independent corpus exists locally, so no new model or accuracy row was
produced. The Stage64 values cannot answer the Stage65B generalization or
causal questions and are retained only as imported controls.
