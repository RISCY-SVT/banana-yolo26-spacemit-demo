#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 PREPARED_PACKAGE OUTPUT_DIR"
  echo "Copies the frozen K1X_INT8_V1 package after exact identity verification."
  exit 0
fi
(( $# == 2 )) || { echo "usage: $0 PREPARED_PACKAGE OUTPUT_DIR" >&2; exit 2; }
source_package=$(cd "$1" && pwd)
output=$2
expected=fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be
actual=$(sha256sum "$source_package/asset_hashes.tsv" | awk '{print $1}')
[[ $actual == "$expected" ]] || { echo "unexpected prepared package identity: $actual" >&2; exit 1; }
mkdir -p "$output"
rsync -a --delete "$source_package/" "$output/"
echo "package_manifest_sha256=$actual"
