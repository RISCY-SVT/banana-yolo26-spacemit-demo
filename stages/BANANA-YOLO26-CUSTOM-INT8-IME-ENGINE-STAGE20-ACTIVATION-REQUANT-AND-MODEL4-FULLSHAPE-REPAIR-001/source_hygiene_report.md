# Source Hygiene Report

status: `pass`

## Checks

```text
git diff --check: pass
git diff --cached --check: pass
symlink scan: pass
secret-like scan: pass, staged files excluding commands.txt self-match logs
changed path ASCII/control scan: pass
large staged file scan: pass
```

The `/data/ncnn` tree had unrelated dirty files at Stage20 preflight. Stage20 did not mutate, clean, inspect deeply, or depend on `/data/ncnn`.

Large full-shape tensor dumps were written under `.deps/custom_int8_engine/stage20_fullshape_oracles/` and were not staged for git.

The final result packet uses sanitized artifacts and is exported through `/data/lab/scripts/export-result-packet.sh`.
