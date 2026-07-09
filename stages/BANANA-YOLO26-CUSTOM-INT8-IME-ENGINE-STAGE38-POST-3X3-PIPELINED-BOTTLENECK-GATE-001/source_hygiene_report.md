# Source Hygiene Report

## Git Diff Checks

- `git diff --check`: `pass`, rc `0`
- `git diff --cached --check`: pending final staged check before commit

## Symlink Scan

Command:

```bash
find custom_int8_engine stages -type l -print
```

Result:

- rc: `0`
- findings: `0`

## Secret-Like Scan

The first xargs-based scan returned `123` because `rg` exits `1` for no matches and xargs maps child non-zero exits to `123`; it had zero findings. The scan was immediately retried with an explicit per-file loop and documented no-match handling.

Final scan result:

- rc: `0`
- findings: `0`
- log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/secret_like_scan_retry.log`

## Path/Control Hygiene

- changed path ASCII/control scan: `pass`, findings `0`
- sensitive path scan: `pass`, findings `0`
- `/control`: not mutated
- `/data/ncnn`: not mutated by this stage
- `/data/banana-yolo11-spacemit-demo`: not touched

## Generated Artifacts

Build directories and board dumps remain outside git-tracked paths. Stage reports and source changes are the only intended tracked changes.
