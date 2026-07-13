#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
mkdir -p "$(dirname "$output")"
"$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --warmup 10 --runs 100 --repeats 5 --verify --benchmark
