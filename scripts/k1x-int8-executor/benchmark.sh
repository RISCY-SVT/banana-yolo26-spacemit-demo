#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
profile=${4:-compatibility}

if [[ -r "$root/config/k1x-int8-executor-stage54.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/config/k1x-int8-executor-stage54.env"
  set +a
fi
case "$profile" in
  compatibility)
    unset Y26_STAGE53_SPIN_POOL
    ;;
  low-latency)
    export Y26_STAGE53_SPIN_POOL=1
    ;;
  *)
    echo "unknown profile: $profile (expected compatibility or low-latency)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
"$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --warmup 10 --runs 100 --repeats 5 --verify --benchmark
