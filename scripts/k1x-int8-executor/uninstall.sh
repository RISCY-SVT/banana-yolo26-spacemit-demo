#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [RELEASE_ROOT]"
  exit 0
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_env=$script_dir/../config/release.env
[[ -f $release_env ]] || release_env=$script_dir/../../config/release.env
# shellcheck disable=SC1090
source "$release_env"
root=${1:-$Y26_BOARD_INSTALL_ROOT}
case "$root" in
  /data/y26-k1x-int8-executor|/data/y26-k1x-int8-executor/*) ;;
  *) echo "refusing unsafe removal root: $root" >&2; exit 2 ;;
esac
if [[ ! -f "$root/release_manifest.json" ]]; then
  echo "release manifest absent; refusing removal: $root" >&2
  exit 2
fi
rm -rf --one-file-system "$root"
