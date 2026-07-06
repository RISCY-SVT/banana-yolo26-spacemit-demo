# source_hygiene_report

## Checks

```text
git diff --check: pass
git diff --cached --check: pass
find custom_int8_engine stages -type l -print: pass, no symlinks
host CTest: pass, 38/38
RISC-V cross build: pass
board correctness: pass, mismatches=0
board stable benchmark: pass, warmup=10 runs=100 repeats=5
```

## Secret-like Scan

Changed/untracked source and stage files were scanned with:

```text
(BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|password=|token=|client_secret|oauth|ssh-rsa|ghp_|glpat-|AKIA|Authorization:)
```

Findings:

```text
commands.txt self-matches for the literal scan command pattern only.
```

No real credentials, private keys, API tokens, or authorization headers were found.

## Path Hygiene

The changed-file path scan found expected local evidence paths:

```text
/home/svt/yolo26-custom-int8-stage24/2026-07-06_19-34-59
/home/svt/.npm-global/bin/codex
/data/.codex/skills/...
```

These appear only in command logs, board-local evidence pointers, and skill-read traceability. They are not runtime source dependencies and are marked as local-only evidence.

No `/control/state/secrets`, `/data/Settings`, credential files, `.env`, SSH keys, or `/data/ncnn` mutation were introduced.

## Artifact Policy

Large generated board binaries, build trees, and full-shape tensor dumps were not staged for git. `.deps/custom_int8_engine/build-*` remains outside tracked source.
