#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

[[ ${1:-} != --help && ${1:-} != -h ]] || {
  echo "usage: $0 [preprocessed-f32 fixture] [extra benchmark arguments]"
  exit 0
}
root=$Y26_BOARD_RELEASE_ROOT
fixture=${1:-$root/outputs/correctness/bus_640_nchw_f32.bin}
if (($#)); then shift; fi
args=("$root/bin/yolo26_k1x_int8" --package "$root/package" --image "$fixture"
      --input-mode preprocessed-f32 --profile low-latency --warmup 10 --runs 100 --repeats 5
      --benchmark --verify-determinism --expected-manifest-sha256 "$Y26_EXPECTED_MANIFEST_SHA256" "$@")
echo "surface=pure_executor_preprocessed_f32"
if y26_is_board; then y26_print_command "${args[@]}"; "${args[@]}"; else y26_remote_command "${args[@]}"; fi
