# K1X INT8 Executor Handoff Checklist

- [ ] Release `SHA256SUMS` validates.
- [ ] Version reports 0.9.2, the frozen full-graph profile, and ABI1 capabilities.
- [ ] Package manifest equals `fab4a72c...e8407be`.
- [ ] Healthcheck returns fixture hash `0xd43f5e018b415631`.
- [ ] Compatibility runs without stage environment variables.
- [ ] Low-latency wake is selected through the CLI/API.
- [ ] O2 dry-run, `run`, and post-run `status` pass when dedicated mode is used.
- [ ] CPU0-3 are IME workers, CPU4 is controller, CPU4-7 IME count is zero.
- [ ] CMake and pkg-config external C examples build and run.
- [ ] Shared SONAME is 1 and only public C ABI symbols are exported.
- [ ] Inputs are already letterboxed into the documented format.
- [ ] Logs and outputs stay on NVMe `/data`.
- [ ] Pure-model and pipeline/throughput statistics are labeled separately.
- [ ] Team understands this is not a 20 FPS or production-certification claim.
