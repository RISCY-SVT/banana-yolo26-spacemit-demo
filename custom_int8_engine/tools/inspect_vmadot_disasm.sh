#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <binary-or-object>" >&2
    exit 2
fi

TARGET="$1"
OBJDUMP="${OBJDUMP:-}"
if [ -z "$OBJDUMP" ]; then
    if command -v riscv64-unknown-linux-gnu-objdump >/dev/null 2>&1; then
        OBJDUMP=riscv64-unknown-linux-gnu-objdump
    elif command -v llvm-objdump >/dev/null 2>&1; then
        OBJDUMP=llvm-objdump
    else
        OBJDUMP=objdump
    fi
fi

"$OBJDUMP" -d "$TARGET" | sed -n '/smt\\.vmadot/,+8p;/vmadot/,+8p'
