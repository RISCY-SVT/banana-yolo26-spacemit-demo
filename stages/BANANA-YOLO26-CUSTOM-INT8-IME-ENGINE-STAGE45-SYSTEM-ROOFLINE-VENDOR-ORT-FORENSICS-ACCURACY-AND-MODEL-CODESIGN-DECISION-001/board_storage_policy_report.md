# Board storage policy

Preflight passed: `/data` is writable ext4 on NVMe (`/dev/nvme0n1p1`) with
`438G` available; `/` is eMMC (`/dev/mmcblk2p6`). All deployed binaries, models,
inputs, profiles, perf failures, logs, temporary files, and caches were kept under
`/data/k1x-stage-runs/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-SYSTEM-ROOFLINE-VENDOR-ORT-FORENSICS-ACCURACY-AND-MODEL-CODESIGN-DECISION-001`. `TMPDIR` and `XDG_CACHE_HOME` were redirected there.

No Stage45 artifact was written to eMMC. The installed ORT library under
`/home/svt` was read in place and not duplicated or modified. Exception count: 0.
