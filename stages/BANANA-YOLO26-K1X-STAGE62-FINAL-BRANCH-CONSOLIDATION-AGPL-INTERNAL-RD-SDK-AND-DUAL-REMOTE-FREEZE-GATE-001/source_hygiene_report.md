# Source Hygiene Report

## Result

Stage62 source and export hygiene passed with the limitations recorded below.

- `git diff --check`: pass.
- `git diff --cached --check`: pass.
- Python tool compilation: pass.
- `bash -n` on changed shell files: pass.
- Tracked symlinks: 0.
- Tracked files larger than 10 MiB: 0.
- Private-key and token-shaped markers: 0.
- Tracked credential filenames: 0.
- Release/result payload symlinks: 0.
- Board project writes to eMMC: 0.
- Large packages, ONNX files, predictions, media, and raw timing data remain under NVMe `/data`, outside Git.

ShellCheck, Syft, ScanCode, and REUSE were unavailable and are recorded as unavailable, not as successful scans. The deterministic assembler generated file, ELF dependency, license, and unresolved-item inventories from the actual release trees.

## `/data/ncnn`

The evidence tree was not modified by Stage62. Its inherited worktree already contains three local edits:

```text
src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

Their latest filesystem modification time is 2026-05-27, before the Stage62 start at 2026-07-22 08:49:39 UTC. Stage62 neither reverted nor changed them. The retained `/data/ncnn` HEAD is `a245a70c641a1f20f357c65d103e5f9e50fe84a1`.

## Raw Evidence

The command outputs are retained under the task shared-log root in `hygiene/final_hygiene.log` and `hygiene/final_hygiene_corrected.log`.
