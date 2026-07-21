# Stage61 Workspace Preflight

- Source branch: `yolo26-k1x-resolution-sweep`
- Source HEAD: `43c02b7051ddde9921a5348e4b4e8986b941212d`
- Stage61 branch: `yolo26-k1x-attention-ntail-r768-q0`
- Start worktree: clean
- GitHub and GitLab source heads: exact source HEAD
- Maintenance 0.9.3 GitHub and GitLab heads: `d0e3611c8d99dfade049bd261cb557509222a456`
- Frozen 0.9.2 branch: `175c1d939cc93fba0e730dba3f1281704e8f25b9`
- Board: Bianbu 2.2.1, Linux 6.6.63
- Board project storage: NVMe `/data`; no project writes to eMMC
- Compiler: SpacemiT GCC 14.3.0, binutils 2.43.1
- Compiler contract: `-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`

The accepted release and frozen branches were read-only references. Stage61 was
created only after exact local and dual-remote parity was established.
