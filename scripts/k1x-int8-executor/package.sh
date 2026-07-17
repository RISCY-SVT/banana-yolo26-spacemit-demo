#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 PREPARED_PACKAGE OUTPUT_DIR"
  echo "Copies the frozen K1X_INT8_V1 package after exact identity verification."
  exit 0
fi
(( $# == 2 )) || { echo "usage: $0 PREPARED_PACKAGE OUTPUT_DIR" >&2; exit 2; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_env=$script_dir/../config/release.env
[[ -f $release_env ]] || release_env=$script_dir/../../config/release.env
# shellcheck disable=SC1090
source "$release_env"
source_package=$(cd "$1" && pwd)
output=$2
expected=$Y26_EXPECTED_MANIFEST_SHA256
actual=$(sha256sum "$source_package/asset_hashes.tsv" | awk '{print $1}')
[[ $actual == "$expected" ]] || { echo "unexpected prepared package identity: $actual" >&2; exit 1; }
mkdir -p "$output"
rsync -a --delete "$source_package/" "$output/"
echo "package_manifest_sha256=$actual"
