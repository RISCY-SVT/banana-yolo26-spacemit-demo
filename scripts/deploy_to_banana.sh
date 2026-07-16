#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

source_root=$Y26_RELEASE_ROOT
destination=$Y26_BOARD_RELEASE_ROOT
usage() { echo "usage: $0 [--source DIR] [--destination /data/PATH]"; }
while (($#)); do
  case $1 in
    --source) source_root=$2; shift 2 ;;
    --destination) destination=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
y26_require_release "$source_root"
[[ $destination == /data/* ]] || { echo "board destination must be under /data" >&2; exit 2; }
ssh "$Y26_BOARD_TARGET" "test -d /data && test -w /data && mkdir -p '$destination'"
rsync -a --delete "$source_root/" "$Y26_BOARD_TARGET:$destination/"
ssh "$Y26_BOARD_TARGET" "cd '$destination' && sha256sum -c SHA256SUMS"
echo "deployed=$Y26_BOARD_TARGET:$destination"
