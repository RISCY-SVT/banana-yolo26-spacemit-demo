#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 TARGET RELEASE_DIR [BOARD_ROOT]"
  exit 0
fi
if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 TARGET RELEASE_DIR [BOARD_ROOT]" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_env=$script_dir/../config/release.env
[[ -f $release_env ]] || release_env=$script_dir/../../config/release.env
# shellcheck disable=SC1090
source "$release_env"
target=$1
release_dir=$(cd "$2" && pwd)
board_root=${3:-$Y26_BOARD_INSTALL_ROOT}
case "$board_root" in
  /data/y26-k1x-int8-executor|/data/y26-k1x-int8-executor/*) ;;
  *) echo "refusing non-NVMe deployment root: $board_root" >&2; exit 2 ;;
esac

ssh "$target" "test -d /data && test -w /data && mkdir -p '$board_root'"
rsync -a --delete --exclude logs/ --exclude outputs/ "$release_dir/" "$target:$board_root/"
ssh "$target" "cd '$board_root' && sha256sum -c SHA256SUMS"
