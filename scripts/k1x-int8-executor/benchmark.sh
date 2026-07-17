#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [RELEASE_ROOT [FIXTURE [OUTPUT_JSON [PROFILE]]]]"
  exit 0
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_env=$script_dir/../config/release.env
[[ -f $release_env ]] || release_env=$script_dir/../../config/release.env
# shellcheck disable=SC1090
source "$release_env"
root=${1:-$Y26_BOARD_INSTALL_ROOT}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
profile=${4:-compatibility}
readonly expected_manifest=$Y26_EXPECTED_MANIFEST_SHA256
case "$profile" in
  compatibility|low-latency|low-latency-dedicated) ;;
  *)
    echo "unknown profile: $profile (expected compatibility, low-latency, or low-latency-dedicated)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
manifest=$(sha256sum "$root/package/asset_hashes.tsv" | awk '{print $1}')
[[ $manifest == "$expected_manifest" ]] || {
  echo "unexpected package manifest: $manifest" >&2
  exit 1
}
command=("$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --profile "$profile" --expected-manifest-sha256 "$expected_manifest" \
  --warmup 10 --runs 100 --repeats 5 --verify-determinism --benchmark)

if [[ $profile != low-latency-dedicated ]]; then
  "${command[@]}"
  exit
fi
"$root/scripts/o2-system-profile.sh" run -- "${command[@]}"
