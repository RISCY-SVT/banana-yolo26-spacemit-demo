# Source Hygiene Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Checks

```text
git diff --check: pass
symlink scan: pass, no symlinks under custom_int8_engine or stages
secret-like scan: pass, no findings
path hygiene scan: documented local-only findings in commands.txt
```

## Path Hygiene Notes

The path scan reported board-local `/home/svt/...` paths inside:

```text
stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/commands.txt
```

These are raw command evidence paths for board deployment and benchmark execution. They are local-only traceability evidence, not portable semantic artifacts. The human-facing reports use sanitized `/data/...` log paths and do not require those board-local paths for interpretation.

## Raw Logs

```text
diff_check_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/final_git_diff_check.log
symlink_scan_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/final_symlink_scan.log
secret_scan_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/final_secret_scan.log
path_scan_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/final_path_scan.log
```

## Forbidden Scope

```text
/data/ncnn mutation by this stage: no
YOLO11 production repo mutation: no
XSlim use: no
vmadotn use: no
CPU4-7 IME execution: no
OpenMP/all-core default dispatch: no
push: no
```
