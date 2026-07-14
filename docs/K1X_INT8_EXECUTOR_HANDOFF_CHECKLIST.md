# K1X INT8 Executor Handoff Checklist

- [ ] Board is Banana-Pi BPI-F3 / SpacemiT K1X with writable NVMe `/data`.
- [ ] Compiler identity and selected ISA/ABI flags match the build policy.
- [ ] Model and package manifest SHA-256 values match the release manifest.
- [ ] Release `sha256sum -c` passes.
- [ ] `ldd` resolves only expected system/release libraries.
- [ ] Package prepare and corruption checks pass.
- [ ] CLI `--version`, preprocessed smoke, image smoke, and `--verify` pass.
- [ ] C API smoke and error paths pass.
- [ ] CPU0-3 worker affinity passes; CPU4-7 IME count is zero.
- [ ] FRM and vector CSR restoration tests pass.
- [ ] Full COCO result and prediction hash match the Stage52 functional-reference
      report and the byte-identical Stage53 optimized-research result.
- [ ] The intended wake policy is explicit: condition-variable compatibility or
      `Y26_STAGE53_SPIN_POOL=1` optimized research.
- [ ] Full-model timing and the corresponding 10,000-run soak match the reported
      statistical surface and wake policy.
- [ ] Optional `rr20` is disabled unless a dedicated lab operator enables it.
- [ ] No ORT, Python, per-run file I/O, or float Q/DQ exists in measured runtime.
- [ ] Deployment is under `/data/k1x-yolo26-int8-executor` or a versioned child
      directory on the same NVMe filesystem.
- [ ] Limitations and rollback/removal instructions were reviewed.
