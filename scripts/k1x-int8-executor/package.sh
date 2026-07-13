#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 REPO MODEL OPTIMIZED_CORE_PACKAGE OUTPUT" >&2
  exit 2
fi
repo=$(cd "$1" && pwd)
model=$2
core=$3
output=$4
python=${Y26_PACKAGE_PYTHON:-"$repo/.deps/venvs/ultralytics_latest/bin/python"}

"$python" "$repo/custom_int8_engine/tools/stage52_full_package.py" \
  --model "$model" --optimized-core-package "$core" --out-dir "$output"
sha256sum "$output/asset_hashes.tsv"
