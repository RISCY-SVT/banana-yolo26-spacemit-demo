#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE_ROOT RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
shift
bin=$stage_root/bin/stage60_resolution_bench
output=$stage_root/long-soak
ram=/dev/shm/y26-stage60-soak-$$
repeats=${Y26_STAGE60_SOAK_REPEATS:-100}
[[ $repeats =~ ^[1-9][0-9]*$ ]] || {
  echo "Y26_STAGE60_SOAK_REPEATS must be a positive integer" >&2
  exit 2
}
mkdir -p "$output" "$ram"

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

sampler_pid=
stop_file=
cleanup() {
  if [[ -n $stop_file ]]; then touch "$stop_file"; fi
  if [[ -n $sampler_pid ]]; then wait "$sampler_pid" 2>/dev/null || true; fi
  rm -rf "$ram"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[[ -x $bin ]] || { echo "missing Stage60 benchmark: $bin" >&2; exit 1; }

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  input=$stage_root/fixtures/r$resolution/bus_r${resolution}_nchw_f32.bin
  prefix=$ram/r${resolution}
  final_prefix=$output/r${resolution}
  stop_file=$prefix.stop
  rm -f "$stop_file"
  printf 'timestamp_utc\tresolution\tmean_thermal_c\tmean_cpu0_4_khz\n' \
    >"$prefix.system.tsv"
  (
    while [[ ! -e $stop_file ]]; do
      temperature=$(awk '{sum += $1; count += 1} END {if (count) printf "%.3f", sum/count/1000}' \
        /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true)
      frequency=$(awk '{sum += $1; count += 1} END {if (count) printf "%.0f", sum/count}' \
        /sys/devices/system/cpu/cpu[0-4]/cpufreq/scaling_cur_freq 2>/dev/null || true)
      printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$resolution" "$temperature" "$frequency" >>"$prefix.system.tsv"
      sleep 5
    done
  ) &
  sampler_pid=$!

  api_begin_ns=$(date +%s%N)
  set +e
  "$bin" --package "$package" --input "$input" --output "$prefix.raw.tsv" \
    --surface preprocessed --wake frame-gated-spin \
    --warmup 10 --runs 100 --repeats "$repeats" \
    >"$prefix.summary.txt" 2>"$prefix.stderr.txt"
  status=$?
  set -e
  api_end_ns=$(date +%s%N)

  touch "$stop_file"
  wait "$sampler_pid" || true
  sampler_pid=
  stop_file=
  printf '%s\n' "$status" >"$prefix.exit-status.txt"
  printf 'resolution\texpected_samples\tprocess_elapsed_ns\tprocess_elapsed_us_per_sample\n' \
    >"$prefix.process-wall.tsv"
  elapsed_ns=$((api_end_ns - api_begin_ns))
  expected_samples=$((100 * repeats))
  printf '%s\t%s\t%s\t%.6f\n' "$resolution" "$expected_samples" "$elapsed_ns" \
    "$(awk -v elapsed="$elapsed_ns" -v samples="$expected_samples" \
        'BEGIN {print elapsed / samples / 1000.0}')" >>"$prefix.process-wall.tsv"
  sha256sum "$bin" "$package/asset_hashes.tsv" "$input" >"$prefix.identities.txt"
  for suffix in raw.tsv summary.txt stderr.txt system.tsv exit-status.txt process-wall.tsv identities.txt; do
    [[ -f $prefix.$suffix ]] && cp "$prefix.$suffix" "$final_prefix.$suffix"
  done
  if [[ $status -ne 0 ]]; then
    echo "Stage60 soak failed for R$resolution" >&2
    exit "$status"
  fi
  samples=$(awk 'END {print NR - 1}' "$final_prefix.raw.tsv")
  if [[ $samples -ne $expected_samples ]]; then
    echo "Stage60 soak row-count mismatch for R$resolution: $samples" >&2
    exit 1
  fi
done
