
# Board storage policy

Board `/data` is writable NVMe (`/dev/nvme0n1p1`, ext4); root is eMMC. All new
Stage48 binaries, packages, outputs, profiles, and logs are under
`/data/k1x-stage-runs/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE48-EXACT-INTEGER-CONTRACT-AND-NCHWC8-DIRECT-A-TILE-PROOF-001`. `TMPDIR` and `XDG_CACHE_HOME` point into that root. The recorded
eMMC exception count is zero.
