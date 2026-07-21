#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE_ROOT RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
shift
binary=$stage_root/bin/stage61_resolution_bench
output=$stage_root/benchmarks/final
ram=/dev/shm/y26-stage61-performance-$$
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

[[ -x $binary ]] || { echo "missing Stage61 benchmark: $binary" >&2; exit 1; }
environment=$ram/environment.tsv
printf 'resolution\tsurface\tphase\tmean_thermal_c\tmean_cpu0_4_khz\tboot_id\n' >"$environment"

snapshot_environment() {
  local resolution=$1 surface=$2 phase=$3
  local temperature frequency
  temperature=$(awk '{sum += $1; count += 1} END {if (count) printf "%.3f", sum/count/1000}' \
    /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true)
  frequency=$(awk '{sum += $1; count += 1} END {if (count) printf "%.0f", sum/count}' \
    /sys/devices/system/cpu/cpu[0-4]/cpufreq/scaling_cur_freq 2>/dev/null || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$resolution" "$surface" "$phase" \
    "$temperature" "$frequency" "$(cat /proc/sys/kernel/random/boot_id)" >>"$environment"
}

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  for surface in preprocessed rgb; do
    if [[ $surface == preprocessed ]]; then
      input=$stage_root/fixtures/r$resolution/bus_r${resolution}_nchw_f32.bin
    else
      input=$stage_root/fixtures/r$resolution/bus_r${resolution}_rgb_u8.bin
    fi
    samples=$ram/r${resolution}_${surface}.tsv
    summary=$ram/r${resolution}_${surface}.summary.txt
    snapshot_environment "$resolution" "$surface" before
    "$binary" --package "$package" --input "$input" --output "$samples" \
      --surface "$surface" --wake frame-gated-spin \
      --warmup 10 --runs 100 --repeats 5 >"$summary" 2>&1
    snapshot_environment "$resolution" "$surface" after
    [[ $(awk 'END {print NR - 1}' "$samples") -eq 500 ]] || {
      echo "Stage61 benchmark row-count mismatch: R$resolution $surface" >&2
      exit 1
    }
    cp "$samples" "$output/r${resolution}_${surface}_raw.tsv"
    cp "$summary" "$output/r${resolution}_${surface}_summary.txt"
  done
done
cp "$environment" "$output/environment.tsv"
