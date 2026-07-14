# Handoff smoke report

The finalized bundle was deployed through `scripts/deploy.sh` to the board
NVMe root `/data/k1x-yolo26-int8-executor`. The deployment-side checksum pass
validated all 1201 payload entries.

The deployed `scripts/smoke-test.sh` then passed:

- CLI version: `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001/abi1`.
- Public C ABI smoke output hash: `d43f5e018b415631`.
- CLI safe-scheduler verification output hash: `d43f5e018b415631`.
- CPU0-3 affinity: pass.
- CPU4-7 IME count: 0.
- Smoke prediction JSON SHA-256:
  `c0068b6f02efe95c6761926587c526ae32fd3e49f95865cf3c159ab61fc38f3b`.

The two-run CLI smoke measured 532842 us median with a 533405 us p95. This
short smoke validates deployment and API behavior; it is not the headline
performance surface. The stable 10/100/5 and 10000-run measurements are in
`full_model_performance_report.md`.

No board artifact was written outside NVMe `/data` by the handoff workflow.
