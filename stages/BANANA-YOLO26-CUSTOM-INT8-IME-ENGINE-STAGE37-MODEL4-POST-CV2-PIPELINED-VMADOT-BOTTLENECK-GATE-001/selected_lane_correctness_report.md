# Selected Lane Correctness Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Candidate

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
CLI: --merge-repair branch3x3_pipelined4
affinity: taskset -c 0-3
```

## Smoke Test

```text
warmup: 1
runs: 1
repeats: 1
status: pass
mismatches: 0
max_abs_diff: 0
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Stable Correctness

```text
warmup: 10
runs: 100
repeats: 5
status: pass
mismatches: 0
max_abs_diff: 0
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
```

## FRM Sweep

```text
ambient RNE: pass, after_frm restored
ambient RTZ: pass, after_frm restored
ambient RDN: pass, after_frm restored
ambient RUP: pass, after_frm restored
ambient RMM: pass, after_frm restored
```

## CPU Policy

```text
IME CPU policy: CPU0-3 only
CPU4-7 IME execution: none
OpenMP/all-core dispatch: not used
```

## Raw Evidence

```text
smoke_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/board_stage37_smoke.log
stable_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/board_stage37_candidate_stable.log
```
