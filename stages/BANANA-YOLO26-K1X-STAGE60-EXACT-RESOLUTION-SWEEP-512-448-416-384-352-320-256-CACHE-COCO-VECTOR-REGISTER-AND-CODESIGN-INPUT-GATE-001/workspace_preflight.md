# Workspace Preflight

## Repository

- Frozen source branch: `yolo26-custom-int8-engine`
- Frozen source HEAD: `175c1d939cc93fba0e730dba3f1281704e8f25b9`
- GitHub frozen-branch HEAD at start: exact frozen source HEAD
- GitLab frozen-branch HEAD at start: exact frozen source HEAD
- Stage60 branch: `yolo26-k1x-resolution-sweep`
- Start worktree: clean
- Pre-stage push: not performed

The Stage60 branch was created directly from the verified frozen commit. The
frozen branch and its 0.9.2 release archives were not modified.

## Toolchain

- Compiler: SpacemiT `riscv64-unknown-linux-gnu-g++` 14.3.0
- Assembler: GNU Binutils 2.43.1.20250119
- Objdump: GNU Binutils 2.43.1.20250119
- ISA/ABI: `-march=rv64gcv_zvfh -mabi=lp64d`
- Tuning: `-mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`
- Sysroot policy: existing read-only base plus K1X GTK3 overlay

## Board

- Target: Banana-Pi BPI-F3 / SpacemiT K1X
- OS: Bianbu 2.2.1
- Kernel: Linux 6.6.63
- Boot ID at preflight: `0a0691d1-7502-44c3-903b-444dba83c1d9`
- CPUs: eight X60 CPUs, with CPU0-3 workers and CPU4 controller
- L2: two 512 KiB cluster caches
- Governor: performance
- Measured CPU frequency: 1.6 GHz
- Stage storage: NVMe `/data`
- eMMC project writes: prohibited

The host and board `/data` mounts were present and writable with sufficient
space. All generated models, packages, predictions, timing samples, builds,
temporary files, and caches use the Stage60 `/data` root.

## Corrected Harness Invocations

Three initial commands were corrected and retained in the raw command ledger:

1. Package generation first used system Python without ONNX, then used the
   accepted Stage20 ONNX virtual environment.
2. The generic release wrapper lacked the explicit prepared cross environment,
   then the release was configured directly with the canonical toolchain file.
3. The first sanitizer configure inherited a RISC-V compiler, then the native
   ASan/UBSan build explicitly selected host GCC/G++.

The corrected commands pass. These were harness/environment errors and did not
produce or select a technical result.

Raw command stdout, stderr, exit codes, and timestamps are in the shared log
directory named in the Stage60 result packet.
