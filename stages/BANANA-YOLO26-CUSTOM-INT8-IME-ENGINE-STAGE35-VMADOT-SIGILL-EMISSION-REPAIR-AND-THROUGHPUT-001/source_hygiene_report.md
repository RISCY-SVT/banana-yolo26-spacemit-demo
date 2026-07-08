# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Checks

```text
git diff --check: pass
find custom_int8_engine stages -type l -print: pass, no symlinks reported
host build: pass
host CTest: pass, 42/42 tests passed
RISC-V cross build with Y26_K1X_ENABLE_IME=ON: pass
board CPU0 SIGILL matrix: pass
board CPU1/2/3 smoke: pass
CPU4-7 IME: not used
```

## Changed Files

Changed-file inventory was written to:

```text
artifacts/changed_files_stage35.txt
```

No large changed files over 1 MiB were reported.

## Secret / Path Scan

Changed-file scan was run over source edits and Stage35 reports.

Expected findings:

```text
commands.txt contains the literal words from the hygiene command itself.
STAGE35_FINAL_REPORT.md initially contained pending scan status before this report updated it.
stage reports contain local evidence paths under /data and board-local /home/svt paths only as evidence references.
source_hygiene_report.md contains literal restricted-path terms only inside a sentence saying those were not found.
```

No SSH private keys, private key bodies, `.env` files, `.git-credentials`, Codex auth/config state, or `/control/state/secrets` references were found.

## Cross-Track Note

Stage35 did not mutate `/data/ncnn`.

## Result

```text
source_hygiene: pass_with_documented_command_log_self_matches
```
