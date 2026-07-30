# System rollback

Stage64 made no boot, kernel, sysctl, service, cgroup, IRQ, workqueue, camera,
or persistent governor change. The boot ID remained
`0a0691d1-7502-44c3-903b-444dba83c1d9`; CPUs 0-7 remained online under the
`performance` governor at 1,600,000 kHz. The global core pattern was not
written.

Fault collection used isolated child processes and one bounded `gdb -batch`
capture. The interrupted CPU profiler stress was terminated with SIGTERM
after preserving its partial evidence. Final inventory found no Stage64
runner, predictor, quantizer, or wrapper process and no Stage64 cgroup.

All models, environments, predictions, profiles, faults, caches, and raw logs
were written under task-local NVMe `/data`. A final targeted scan found zero
Stage64-named artifacts under `/tmp`, `/var/tmp`, or `/home/svt` on the board.
The eMMC project-write exception count is zero.

All four protected Stage62 release roots retain their accepted
`release_manifest.json`, `SHA256SUMS`, and `release_tree_manifest.tsv`
identities. `/data/ncnn` remains at
`a245a70c641a1f20f357c65d103e5f9e50fe84a1` with the same three pre-existing
modified files and hashes recorded at preflight.

Rollback status: **pass; no global rollback was required**.
