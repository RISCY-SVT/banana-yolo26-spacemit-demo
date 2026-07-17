#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [RELEASE_ROOT [FIXTURE [OUTPUT_JSON]]]"
  exit 0
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_env=$script_dir/../config/release.env
[[ -f $release_env ]] || release_env=$script_dir/../../config/release.env
# shellcheck disable=SC1090
source "$release_env"
root=${1:-$Y26_BOARD_INSTALL_ROOT}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/smoke.json"}
readonly expected_manifest=$Y26_EXPECTED_MANIFEST_SHA256
mkdir -p "$(dirname "$output")"
"$root/bin/yolo26_k1x_int8" --version
"$root/bin/y26_k1x_healthcheck" --build-info
manifest=$(sha256sum "$root/package/asset_hashes.tsv" | awk '{print $1}')
[[ $manifest == "$expected_manifest" ]] || {
  echo "unexpected package manifest: $manifest" >&2
  exit 1
}
"$root/bin/y26_k1x_healthcheck" \
  "$root/package" "$expected_manifest" "$input" "${Y26_EXPECTED_OUTPUT_HASH#0x}"
"$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --profile compatibility \
  --threads 4 --pin 0-3 --scheduler safe \
  --expected-manifest-sha256 "$expected_manifest" \
  --verify-determinism --verify-known-fixture \
  --expected-output-hash "$Y26_EXPECTED_OUTPUT_HASH" \
  --warmup 1 --runs 2 --repeats 1 --benchmark
sha256sum "$output"
