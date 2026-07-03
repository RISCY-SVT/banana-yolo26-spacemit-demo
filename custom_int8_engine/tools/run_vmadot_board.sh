#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <board-probe-binary> [bench-binary]" >&2
    exit 2
fi

TARGET="${BANANA_SSH_TARGET:-${BANANA_SSH:-svt@banana}}"
REMOTE_DIR="${REMOTE_DIR:-/home/svt/contcodex/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001}"
PROBE="$1"
BENCH="${2:-}"

ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"
scp "$PROBE" "$TARGET:$REMOTE_DIR/test_vmadot_4x4x8_board_probe"
if [ -n "$BENCH" ]; then
    scp "$BENCH" "$TARGET:$REMOTE_DIR/bench_vmadot_microkernel"
fi

for cpu in 0 1 2 3; do
    echo "=== cpu${cpu} probe ==="
    ssh "$TARGET" "timeout 10s taskset -c $cpu '$REMOTE_DIR/test_vmadot_4x4x8_board_probe'"
done

if [ -n "$BENCH" ]; then
    echo "=== cpu0 microbench ==="
    ssh "$TARGET" "timeout 20s taskset -c 0 '$REMOTE_DIR/bench_vmadot_microkernel' 10000 5"
fi
