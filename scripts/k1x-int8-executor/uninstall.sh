#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [RELEASE_ROOT]"
  exit 0
fi
root=${1:-/data/y26-k1x-int8-executor/0.9.1}
case "$root" in
  /data/y26-k1x-int8-executor|/data/y26-k1x-int8-executor/*) ;;
  *) echo "refusing unsafe removal root: $root" >&2; exit 2 ;;
esac
if [[ ! -f "$root/release_manifest.json" ]]; then
  echo "release manifest absent; refusing removal: $root" >&2
  exit 2
fi
rm -rf --one-file-system "$root"
