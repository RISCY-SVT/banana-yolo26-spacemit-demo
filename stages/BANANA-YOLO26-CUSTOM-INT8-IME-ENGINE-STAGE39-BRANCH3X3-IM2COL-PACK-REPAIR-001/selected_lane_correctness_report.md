# Selected Lane Correctness Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


- selected_mode: `Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK + Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE`
- mismatches: `0`
- max_abs_diff: `0`
- output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- affinity_ok: `1`
- CPU4-7 IME execution: none in selected path; IME workers remain CPU0-3 only.

Actual output SHA evidence:

```text
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433  stage39_stage38_replay_actual.bin
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433  stage39_fastpack_actual.bin
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433  stage39_stage38_replay_after_actual.bin
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433  stage39_stage38_no_measure_actual.bin
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433  stage39_fastpack_no_measure_actual.bin

```
