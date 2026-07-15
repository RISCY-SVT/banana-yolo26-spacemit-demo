#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/smoke.json"}
profile="$root/config/k1x-int8-executor-stage55.env"
if [[ ! -r "$profile" ]]; then
  profile="$root/config/k1x-int8-executor-stage54.env"
fi
if [[ -r "$profile" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$profile"
  set +a
fi
unset Y26_STAGE53_SPIN_POOL Y26_STAGE55_FRAME_GATED_SPIN
mkdir -p "$(dirname "$output")"
"$root/bin/yolo26_k1x_int8" --version
manifest=$(sha256sum "$root/package/asset_hashes.tsv" | awk '{print $1}')
"$root/bin/y26_k1x_c_api_smoke" \
  "$root/package" "$manifest" "$input" d43f5e018b415631
"$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --warmup 1 --runs 2 --repeats 1 --verify --benchmark
sha256sum "$output"
