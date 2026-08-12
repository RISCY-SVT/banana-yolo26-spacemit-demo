# FP32 split and harness decision

Classification: `imported-fp32-surface-confounded`.

The current runner produced byte-identical F0/F1 raw outputs for all 500 H500
images and all 5000 val2017 images. F1 and H8 were also byte-identical on both
surfaces. All three current full-val prediction files have SHA-256
`b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a` and mAP50-95
`0.4018217950262668`. The older imported Stage64 FP32 prediction
surface differs, so it is not evidence of a split residual.
