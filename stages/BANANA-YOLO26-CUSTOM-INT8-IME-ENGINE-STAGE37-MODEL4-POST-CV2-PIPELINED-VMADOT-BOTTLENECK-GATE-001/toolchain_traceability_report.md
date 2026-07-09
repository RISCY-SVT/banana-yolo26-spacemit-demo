# Toolchain Traceability Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Host

```text
host compiler: /usr/bin/g++
host build dir: .deps/custom_int8_engine/build-host-native-stage37
Y26_K1X_ENABLE_IME: OFF
host ctest: 42/42 passed
```

## RISC-V Cross

```text
cross compiler: /opt/riscv/bin/riscv64-unknown-linux-gnu-g++
resolved route: /opt/riscv -> /opt/SpacemiT
build dir: .deps/custom_int8_engine/build-riscv-stage37
Y26_K1X_ENABLE_IME: ON
target flags: -march=rv64gcv_zvfh -mabi=lp64d
cross build: pass
```

## Objdump

```text
objdump: riscv64-unknown-linux-gnu-objdump
objdump_status: pass
evidence: run_logs/objdump_stage37_vmadot.log
```

## Raw Logs

```text
preflight_toolchain_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/preflight_toolchain_which.log
host_build_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/host_build_after_stage37_candidate.log
host_ctest_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/host_ctest_after_stage37_candidate.log
riscv_build_log: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/riscv_build_after_stage37_candidate.log
```
