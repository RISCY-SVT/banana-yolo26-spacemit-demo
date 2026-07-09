# ONNX Cut Gate Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


The Stage39 candidate was run through the real selected `/model.4` ONNX-cut runner path. The dumped output matched the accepted same-input ONNX cut SHA.

- expected_output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- candidate_output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- mismatches: `0`
- max_abs_diff: `0`
- checksum: `106597930`
- expected_checksum: `106597930`

This remains selected-subgraph evidence only, not full YOLO26 inference.
