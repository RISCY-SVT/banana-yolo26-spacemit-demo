# Workspace preflight

- Repository: `/data/banana-yolo26-spacemit-demo`
- Branch: `yolo26-custom-int8-engine`
- Expected and actual start HEAD: `b163c83f7dc1677c8b31b9a2cc75e227d5992b0d`
- Initial worktree: clean
- `git diff --check`: pass
- `git diff --cached --check`: pass
- Board NVMe `/data`: writable ext4 on `/dev/nvme0n1p1`; root is eMMC `/dev/mmcblk2p6`
- `/data/ncnn`: observed only; its pre-existing dirty paths and hashes were recorded and preserved.
