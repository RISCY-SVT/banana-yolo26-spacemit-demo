# Source Hygiene Report

## Checks

| check | status | notes |
|---|---|---|
| `git diff --check` | pass | no whitespace errors |
| host CTest | pass | 34/34 |
| RISC-V cross build | pass | `.deps/custom_int8_engine/build-riscv-stage18` |
| board CPU0-3 smoke | pass | RNE regression and Stage16 runner |
| Stage18 threaded correctness | pass | 1/2/3/4 threads, mismatches=0 |
| Stage18 stable microbench | pass | warmup=10 runs=100 repeats=5 |
| symlink scan | pass | no symlinks under `custom_int8_engine` or `stages` |
| changed path ASCII/control scan | pass | no non-ASCII/control/backslash paths |
| changed-file secret-like scan | pass | no matches outside command-log self patterns |
| large changed files | pass | no large binary/model/dump artifacts |

## External Dirty Trees

`/data/ncnn` showed pre-existing source changes during hygiene:

```text
src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

Stage18 did not mutate `/data/ncnn`.

`/data/banana-yolo11-spacemit-demo` showed unrelated untracked local files:

```text
.claude/
AGENTS.md
```

Stage18 did not mutate the YOLO11 production repo.

## Scope Confirmation

```text
full_engine_implemented: false
graph_wide_scheduler_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
sliding_vmadot123_used: false
CPU4-7 IME used: false
```
