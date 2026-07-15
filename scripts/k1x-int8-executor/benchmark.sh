#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
profile=${4:-compatibility}

operator_profile="$root/config/k1x-int8-executor-stage55.env"
if [[ ! -r "$operator_profile" ]]; then
  operator_profile="$root/config/k1x-int8-executor-stage54.env"
fi
if [[ -r "$operator_profile" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$operator_profile"
  set +a
fi
case "$profile" in
  compatibility)
    unset Y26_STAGE53_SPIN_POOL Y26_STAGE55_FRAME_GATED_SPIN
    ;;
  low-latency)
    export Y26_STAGE53_SPIN_POOL=1
    export Y26_STAGE55_FRAME_GATED_SPIN=1
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
