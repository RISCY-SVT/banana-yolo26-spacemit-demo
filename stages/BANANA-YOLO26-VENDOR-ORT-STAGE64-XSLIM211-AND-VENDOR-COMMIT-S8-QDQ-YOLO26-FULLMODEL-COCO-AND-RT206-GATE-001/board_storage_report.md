# Board storage report

Stage64 project artifacts are rooted at:

```text
/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001
```

At preflight, `/data` was `/dev/nvme0n1p1` (ext4), with 409.7 GiB
available. The root filesystem was `/dev/mmcblk2p6`; it was not selected for
models, predictions, profiles, cores, logs, caches, or temporary files.

Task-local `TMPDIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`, and
`CCACHE_DIR` are under the Stage64 root. Two early bounded host/board smoke
runs predated the explicit environment wrapper. A post-run scan found no
Stage64 project artifacts in `/tmp` or `/var/tmp`; this is recorded as a
procedural caveat, not as an observed eMMC project write.

At closure the board task tree contains 756 files and 1,483,301,120 bytes on
NVMe. The final project process count, Stage64 cgroup count, and matching eMMC
artifact count are all zero. Board eMMC project-write exceptions: **0**.
