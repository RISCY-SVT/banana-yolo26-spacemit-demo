#!/usr/bin/env bash
set -euo pipefail

skill="${1:-$(cd "$(dirname "$0")" && pwd)/SKILL.md}"
test -f "$skill"
test "$(sed -n '1p' "$skill")" = "---"
test "$(rg -n '^---$' "$skill" | wc -l)" -ge 2
rg -q '^name: k1x-board-storage-policy$' "$skill"
rg -q '^## Preflight$' "$skill"
rg -q '^## Stage-Owned Root$' "$skill"
rg -q '^## eMMC Exceptions$' "$skill"
rg -q '^## Cleanup Safety$' "$skill"
rg -q '/data/k1x-stage-runs/' "$skill"
! rg -q 'BEGIN (OPENSSH|RSA|PRIVATE) KEY|password=|client_secret|token=' "$skill"
printf 'k1x_board_storage_policy_selftest=pass\n'
