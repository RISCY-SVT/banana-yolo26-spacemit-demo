#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
case "$root" in
  /data/k1x-yolo26-int8-executor|/data/k1x-yolo26-int8-executor/*) ;;
  *) echo "refusing unsafe removal root: $root" >&2; exit 2 ;;
esac
if [[ ! -f "$root/release_manifest.json" ]]; then
  echo "release manifest absent; refusing removal: $root" >&2
  exit 2
fi
rm -rf --one-file-system "$root"
