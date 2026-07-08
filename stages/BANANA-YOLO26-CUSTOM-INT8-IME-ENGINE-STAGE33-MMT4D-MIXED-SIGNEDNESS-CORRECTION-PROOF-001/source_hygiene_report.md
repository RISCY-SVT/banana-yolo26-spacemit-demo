# Source Hygiene Report

status: `pass`

Final hygiene commands recorded before commit:

```bash
git diff --check
git diff --cached --check
find custom_int8_engine stages -type l -print
git status --short
```

Secret/path scan policy:

```text
scan changed tracked files only
exclude no files by default
document expected rg exit code 1 for no findings
```

Results:

```text
git diff --check: pass
find custom_int8_engine stages -type l -print: no symlinks found
changed-file secret-like scan: no findings
changed-file path scan: board-local evidence path findings only
```

Path scan findings:

```text
/home/svt/yolo26-custom-int8-stage33
```

This is the board-side deployment directory recorded in `commands.txt`. It is local execution evidence, not a secret and not a host credential path.

Raw logs:

```text
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/final_git_diff_check.log
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/final_symlink_scan.log
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/final_secret_scan.log
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/final_path_scan.log
```
