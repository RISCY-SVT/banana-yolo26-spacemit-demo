# Board IME Correctness Report

board_cluster0_tests: pass

## Board surface

- target: `svt@banana`
- kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
- run policy: CPU0-3 only via `taskset -c`
- CPU4-7 negative probe: not run in Stage 1
- remote dir: `/home/svt/contcodex/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001`

## Probe binary

- local path: `.deps/custom_int8_engine/build-k1x-vmadot/tests/test_vmadot_4x4x8_board_probe`
- binary type: `ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI`
- deployed file: `/home/svt/contcodex/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001/test_vmadot_4x4x8_board_probe`

## Results

Each CPU0-3 run reported:

```text
ime_buildtime_available=1
case status mismatches checksum_scalar checksum_ime
all_zeros 0 0 0 0
all_ones 0 0 128 128
ramp 0 0 -640 -640
alternating_edges 0 0 -2080768 -2080768
random_seed_0 0 0 111883 111883
random_seed_1 0 0 17217 17217
random_seed_12345 0 0 -15904 -15904
accumulate_true 0 0 -15912 -15912
total_mismatches=0
nonzero_status=0
```

CPU affinity observed:

- CPU0: `cpu_before=0`, `cpu_after=0`
- CPU1: `cpu_before=1`, `cpu_after=1`
- CPU2: `cpu_before=2`, `cpu_after=2`
- CPU3: `cpu_before=3`, `cpu_after=3`

Raw logs:

- `$LOG_DIR/run_logs/board_probe_retry_cpu0.stdout`
- `$LOG_DIR/run_logs/board_probe_retry_cpu1.stdout`
- `$LOG_DIR/run_logs/board_probe_retry_cpu2.stdout`
- `$LOG_DIR/run_logs/board_probe_retry_cpu3.stdout`
