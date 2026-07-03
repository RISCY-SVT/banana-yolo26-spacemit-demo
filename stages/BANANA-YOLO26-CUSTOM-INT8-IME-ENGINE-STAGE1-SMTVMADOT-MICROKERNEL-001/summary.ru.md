# Stage 1 Summary

classification: stage1-vmadot-microkernel-board-proven-ready-for-conv-stage
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001

Stage 1 выполнен в узком объеме: добавлен только `smt.vmadot 4x4x8 s8xs8->s32` microkernel, scalar oracle, deterministic tests, board probe, disassembly helper и bounded microbench. Полный YOLO26 engine, graph scheduler, Conv integration, camera demo, full COCO и production artifacts не реализовывались.

Доказано:

- host scalar-only build с явным `/usr/bin/g++`: `ctest` 8/8 pass;
- RISC-V cross-build через SpacemiT toolchain: pass;
- named asm route: `smt.vmadot`, raw `.insn` не использовался;
- disassembly содержит `smt.vmadot v28,v0,v1`;
- board CPU0, CPU1, CPU2, CPU3: все deterministic vectors прошли, `status=0`, `mismatches=0`;
- CPU4-7 не запускались;
- direct benchmark-only IME body на CPU0 показал `direct_speedup_vs_scalar=8.058` для tiny 4x4x8 tile; это не YOLO26 FPS и не model-level benchmark.

Ограничения:

- zero-point correction и requantization остаются за Stage 2+;
- public guarded API включает `sched_getcpu` и SIGILL guard на каждый вызов, поэтому его per-call benchmark медленнее scalar;
- `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn`, FP/vfmadot не реализованы;
- `xslim` не использовался.

Следующий рекомендуемый шаг: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE2-CONV1X1-3X3-KERNEL-BRINGUP-001` после review/approval.
