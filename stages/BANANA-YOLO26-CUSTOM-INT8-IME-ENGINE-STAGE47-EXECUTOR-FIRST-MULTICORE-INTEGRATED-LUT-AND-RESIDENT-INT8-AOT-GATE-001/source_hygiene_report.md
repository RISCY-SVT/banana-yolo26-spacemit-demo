
# Source hygiene

Host build and 45/45 CTest pass; focused ASan/UBSan passes; RISC-V cross-build,
board loader, FRM sweep, CPU0-3 affinity, and Python compile pass. Final Git,
symlink, large-file, and secret/path scans are recorded in the shared command
ledger. No models, tensor dumps, build trees, board logs, datasets, vendor
runtimes, credentials, or symlinks are included in the intended commit.
