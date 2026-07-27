# Workspace preflight

- Protected main, stable tag, and integrated tag matched the authorized SHAs
  locally and on both remotes before the research worktree was created.
- The source worktree was clean. The Stage63 branch was created from
  `1fd2e71bb1d5a924e7c0444cada94f681b73aa91`.
- Stage63 changes are confined to `vendor_ort_validation/` and this report
  directory. `custom_int8_engine/` and accepted release roots are read-only.
- Host artifacts are on `/data` NVMe. Board artifacts are under the task-owned
  `/data/k1x-stage-runs/...` root.
- `/data/ncnn` had three pre-existing modified convolution files at preflight;
  Stage63 does not touch that repository.
- Compiler: SpacemiT GCC 14.3.0 (`g56971dcbea2`).
- Binutils: 2.43.1.20250119.
- Board: Bianbu 2.2.1, Linux 6.6.63, eight X60 CPUs, performance governor,
  1.6 GHz, boot ID `0a0691d1-7502-44c3-903b-444dba83c1d9`.

Raw command output is retained in the task log root as
`preflight_host.log` and `preflight_board_readonly.log`.
