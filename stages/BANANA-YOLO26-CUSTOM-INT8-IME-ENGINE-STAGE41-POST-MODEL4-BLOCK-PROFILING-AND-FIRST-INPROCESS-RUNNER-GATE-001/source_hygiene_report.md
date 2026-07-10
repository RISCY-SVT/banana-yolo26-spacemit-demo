# Source Hygiene Report

Preflight:

```text
git diff --check: pass
git diff --cached --check: pass
symlink scan: no findings
```

Build and test:

```text
host build: pass
host CTest: pass, 42/42
RISC-V cross build with Y26_K1X_ENABLE_IME=ON: pass
board selected-mode in-process runner: executed, affinity_ok=1, correctness fail due ORT contract mismatch
```

Final scan:

```text
git diff --check: pass
git diff --cached --check: pass
symlink scan: no findings
changed/untracked secret/path scan: no findings
rg no-match exit code: expected and documented in /data/ncnn-logs/ai-team/2026-07-09_17-04-15/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE41-POST-MODEL4-BLOCK-PROFILING-AND-FIRST-INPROCESS-RUNNER-GATE-001/run_logs/final_hygiene_all_changed.txt
```

Commit policy:

```text
local commit created: no
reason: hard board selected-mode full-output gate failed, so Stage41 is classified as partial correctness only
end_head: unchanged at 6559e2a4a146e96df9db37bf748808896d08e147
```

Cross-track state:

```text
/data/ncnn had unrelated dirty files in src/layer/riscv/convolution_1x1_int8_xsmtvdot.*; Stage41 did not touch /data/ncnn.
/data/banana-yolo11-spacemit-demo was inspected only and not mutated.
```

Export policy:

```text
raw logs remain under /data/ncnn-logs/ai-team/...
safe reports and summaries are exported through /data/lab/scripts/export-result-packet.sh
credentials, .env files, SSH keys, and Codex config are not exported
```

Final validation details are recorded in the result packet raw evidence paths.
