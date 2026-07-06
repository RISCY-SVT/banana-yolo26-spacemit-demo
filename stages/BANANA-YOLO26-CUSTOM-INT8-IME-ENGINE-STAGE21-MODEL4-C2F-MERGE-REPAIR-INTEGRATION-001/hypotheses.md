# Stage21 Pre-Registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6ea3f0737c2063de94a7b4beac976180c4375872`
previous_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001`
selected_repair_from_stage20: `C2_split0_concat_lut_4t`

## H1

The current real model4 C2f runner merge path still performs the split0/concat handling in a way equivalent to the Stage20 B1 baseline, so the Stage20 C2 repair can transfer to the engine path. This must be confirmed by source inspection and baseline timing before claiming performance transfer.

## H2

Integrating `C2_split0_concat_lut_4t` into the real model4 C2f runner preserves exact output: `concat_mismatches=0` and `model4_cv2_mismatches=0` against the ONNX CPU oracle.

## H3

Integrated full-shape model4 C2f C2 timing remains within +3% of the Stage20 C2 sidecar mean total time, `<= 119828 us`, under the stable benchmark protocol. If it is between +3% and +10%, `<= 127972 us`, Stage21 must classify as warning and explain transfer overhead. If it exceeds +10%, Stage21 must classify as integration performance failure and must not close as accepted.
