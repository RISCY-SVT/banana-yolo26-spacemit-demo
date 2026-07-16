#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
profile=${4:-compatibility}
case "$profile" in
  compatibility|low-latency|low-latency-dedicated) ;;
  *)
    echo "unknown profile: $profile (expected compatibility, low-latency, or low-latency-dedicated)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
manifest=$(sha256sum "$root/package/asset_hashes.tsv" | awk '{print $1}')
command=("$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --profile "$profile" --expected-manifest-sha256 "$manifest" \
  --warmup 10 --runs 100 --repeats 5 --verify-determinism --benchmark)

if [[ $profile != low-latency-dedicated ]]; then
  "${command[@]}"
  exit
fi
"$root/scripts/o2-system-profile.sh" run -- "${command[@]}"
