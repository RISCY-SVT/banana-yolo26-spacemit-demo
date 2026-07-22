#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE_ROOT RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
shift
binary=$stage_root/bin/stage61_resolution_bench
output=$stage_root/performance/route-counts
ram=/dev/shm/y26-stage61-route-counts-$$
mkdir -p "$output" "$ram"
trap 'rm -rf "$ram"' EXIT

export Y26_STAGE54_E2C3=1
export Y26_STAGE55_E2C4=1
export Y26_STAGE55_DENSE_FAMILY_A=1
export Y26_STAGE54_DIRECT_1X1=1
export Y26_STAGE54_DENSE_PACK_RVV=1
export Y26_STAGE53_FUSED_LUT=1
export Y26_STAGE54_DEPTHWISE_V2=1
export Y26_STAGE54_DEPTHWISE_X2=1
export Y26_STAGE54_DEPTHWISE_BORDER_V2=1
export Y26_STAGE54_INPUT_RVV_V2=1
export Y26_STAGE54_INPUT_COMPACT_C3=1
export Y26_STAGE54_LUT2_RVV=1
export Y26_STAGE54_ATTENTION_V2=1
export Y26_STAGE56_HEAD_PRODUCER_REDUCTION=1
export Y26_STAGE56_ATTENTION_DIRECT_PACK=1
export Y26_STAGE57_E2C5=1
export Y26_STAGE57_ATTENTION_MATMUL_C8=1
export Y26_STAGE57_RGB_COPY_RVV=1
export Y26_STAGE61_ATTENTION_NTAIL=1
export Y26_STAGE61_ROUTE_COUNTS=1

printf 'resolution\tn4_count\tn8_count\tn16_count\tpadded_dead_columns\tk_padding_lanes\tscalar_fallback_count\toutput_hash\tpackage_manifest_sha256\tcpu4_7_ime_count\n' \
  >"$ram/route-counts.tsv"

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  input=$stage_root/fixtures/r$resolution/bus_r${resolution}_nchw_f32.bin
  prefix=$ram/r$resolution
  "$binary" --package "$package" --input "$input" --output "$prefix.samples.tsv" \
    --surface preprocessed --wake frame-gated-spin --warmup 0 --runs 1 --repeats 1 \
    >"$prefix.stdout" 2>"$prefix.stderr"
  route=$(grep '^stage61_attention_routes' "$prefix.stderr" | tail -n 1)
  [[ -n $route ]] || { echo "missing route counts for R$resolution" >&2; exit 1; }
  n4=$(sed -n 's/.*\tn4=\([0-9]*\).*/\1/p' <<<"$route")
  n8=$(sed -n 's/.*\tn8=\([0-9]*\).*/\1/p' <<<"$route")
  n16=$(sed -n 's/.*\tn16=\([0-9]*\).*/\1/p' <<<"$route")
  dead=$(sed -n 's/.*\tpadded_dead_columns=\([0-9]*\).*/\1/p' <<<"$route")
  kpad=$(sed -n 's/.*\tk_padding_lanes=\([0-9]*\).*/\1/p' <<<"$route")
  scalar=$(sed -n 's/.*\tscalar_fallbacks=\([0-9]*\).*/\1/p' <<<"$route")
  read -r output_hash manifest cpu4_7 < <(
    awk -F '\t' 'NR == 2 {print $21, $22, $20}' "$prefix.samples.tsv"
  )
  [[ $scalar -eq 0 && $cpu4_7 -eq 0 ]] || {
    echo "invalid Stage61 route ownership for R$resolution" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$resolution" "$n4" "$n8" "$n16" "$dead" "$kpad" "$scalar" \
    "$output_hash" "$manifest" "$cpu4_7" >>"$ram/route-counts.tsv"
  cp "$prefix.stderr" "$output/r${resolution}.stderr"
  cp "$prefix.samples.tsv" "$output/r${resolution}.samples.tsv"
done

cp "$ram/route-counts.tsv" "$output/attention_ntail_route_counts.tsv"
