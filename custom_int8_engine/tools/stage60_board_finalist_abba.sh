#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 STAGE_ROOT" >&2
  exit 2
fi

stage_root=$1
runner=$stage_root/bin/stage60_resolution_bench
output=$stage_root/benchmarks/finalist-abba
ram=/dev/shm/y26-stage60-finalist-abba-$$
mkdir -p "$output" "$ram"
trap 'rm -rf "$ram"' EXIT

[[ -x $runner ]] || { echo "missing Stage60 runner: $runner" >&2; exit 1; }

# Freeze the accepted 0.9.2 operator route. Resolution is the only variable.
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

run_comparison() {
  local candidate=$1
  local result=$output/r640_vs_r${candidate}.tsv
  local sequence=(640 "$candidate" "$candidate" 640 "$candidate" 640 640 "$candidate" 640 "$candidate")
  local index=0

  : >"$result"
  for resolution in "${sequence[@]}"; do
    index=$((index + 1))
    local raw=$ram/r${resolution}-position${index}.tsv
    "$runner" \
      --package "$stage_root/packages/r$resolution" \
      --input "$stage_root/fixtures/r$resolution/bus_r${resolution}_nchw_f32.bin" \
      --output "$raw" --surface preprocessed --wake frame-gated-spin \
      --warmup 10 --runs 100 --repeats 1 \
      >"$ram/r${resolution}-position${index}.summary.txt" 2>&1
    if [[ $index -eq 1 ]]; then
      printf 'comparison\tsequence_index\tarm\t' >>"$result"
      head -n 1 "$raw" >>"$result"
    fi
    awk -v comparison="r640-vs-r${candidate}" -v position="$index" \
        -v arm="r${resolution}" 'NR > 1 {print comparison "\t" position "\t" arm "\t" $0}' \
        "$raw" >>"$result"
  done
  [[ $(wc -l <"$result") -eq 1001 ]] || {
    echo "incomplete finalist comparison for R${candidate}" >&2
    return 1
  }
}

run_comparison 512
run_comparison 384
