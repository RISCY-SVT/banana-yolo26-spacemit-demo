# Released XSlim configuration contract

Binding: published xslim-2.1.2+riscy.1-py3-none-any.whl, SHA-256
635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784.

The released package defines:

- calibration_step: requested calibration count; default 500 and accepted
  bounded range 10 through 1000.
- calibration_type: default, minmax, percentile, kl, or mse; this matrix uses
  default.
- precision_level: policy enum 0 through 4 (plus 100); level 1 may retain
  sensitive regions at higher precision.
- finetune_level: level 1 for the Stage64 reproduction control and level 2
  for the released detector starting policy. Level 2 fine-tunes the top five
  selected blocks.
- analysis_enable=true: emits Graphwise diagnostics.
- truncate_var_names: the six accepted YOLO26 bbox/confidence boundary names;
  all lanes use the same ordered list.
- custom_setting: supported by the released package but intentionally not
  used in this baseline factor matrix.

The released accuracy guide recommends 500 to 1000 images for detection and
P1/F2 as the YOLO starting policy. Its diagnostic thresholds are SNR >= 0.1
for high error and cosine < 0.99 for significant deviation.

XSlim's released implementation also samples equalization data with Python
random.sample and shuffles F2 training steps without a CLI seed. Decision runs
therefore use the unchanged released wheel through the stage-local launcher
stage65b_r1_seeded_xslim.py, seed 65001, and deterministic Torch algorithms.
The abandoned unseeded preflight logs are retained separately.
