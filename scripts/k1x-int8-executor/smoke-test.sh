#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/smoke.json"}
mkdir -p "$(dirname "$output")"
"$root/bin/yolo26_k1x_int8" --version
manifest=$(sha256sum "$root/package/asset_hashes.tsv" | awk '{print $1}')
"$root/bin/y26_k1x_healthcheck" \
  "$root/package" "$manifest" "$input" d43f5e018b415631
"$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --profile compatibility \
  --threads 4 --pin 0-3 --scheduler safe \
  --expected-manifest-sha256 "$manifest" \
  --verify-determinism --verify-known-fixture \
  --expected-output-hash 0xd43f5e018b415631 \
  --warmup 1 --runs 2 --repeats 1 --benchmark
sha256sum "$output"
