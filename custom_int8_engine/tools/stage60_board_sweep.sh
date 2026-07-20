#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STAGE_ROOT performance|shape-matrix" >&2
  exit 2
fi

stage_root=$1
mode=$2
bin=$stage_root/bin
ram=/dev/shm/y26-stage60-$mode-$$
mkdir -p "$ram"
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

snapshot_environment() {
  local resolution=$1 surface=$2 phase=$3 output=$4
  local temperature frequency
  temperature=$(awk '{sum += $1; count += 1} END {if (count) printf "%.3f", sum/count/1000}' \
    /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true)
  frequency=$(awk '{sum += $1; count += 1} END {if (count) printf "%.0f", sum/count}' \
    /sys/devices/system/cpu/cpu[0-4]/cpufreq/scaling_cur_freq 2>/dev/null || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$resolution" "$surface" "$phase" "$temperature" "$frequency" \
    "$(cat /proc/sys/kernel/perf_event_paranoid)" \
    "$(cat /proc/sys/kernel/random/boot_id)" >>"$output"
}

case "$mode" in
  performance)
    output=$stage_root/benchmarks
    mkdir -p "$output"
    environment=$ram/environment.tsv
    printf 'resolution\tsurface\tphase\tmean_thermal_c\tmean_cpu0_4_khz\tperf_event_paranoid\tboot_id\n' \
      >"$environment"
    for resolution in 640 512 448 416 384 352 320 256; do
      package=$stage_root/packages/r$resolution
      for surface in preprocessed rgb; do
        if [[ $surface == preprocessed ]]; then
          input=$stage_root/fixtures/r$resolution/bus_r${resolution}_nchw_f32.bin
        else
          input=$stage_root/fixtures/r$resolution/bus_r${resolution}_rgb_u8.bin
        fi
        samples=$ram/r${resolution}_${surface}.tsv
        summary=$ram/r${resolution}_${surface}.summary.txt
        snapshot_environment "$resolution" "$surface" before "$environment"
        "$bin/stage60_resolution_bench" \
          --package "$package" --input "$input" --output "$samples" \
          --surface "$surface" --wake frame-gated-spin \
          --warmup 10 --runs 100 --repeats 5 >"$summary" 2>&1
        snapshot_environment "$resolution" "$surface" after "$environment"
        cp "$samples" "$output/r${resolution}_${surface}_raw.tsv"
        cp "$summary" "$output/r${resolution}_${surface}_summary.txt"
      done
    done
    cp "$environment" "$output/resolution_benchmark_environment.tsv"
    ;;
  shape-matrix)
    output=$stage_root/dense
    mkdir -p "$output"
    export Y26_STAGE60_SHAPE_OPERATIONS=5,12,124,133,144,161,201,206
    for resolution in 384 352 320; do
      package=$stage_root/packages/r$resolution
      manifest=$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')
      for route in m12n16 m8n16 m4tail; do
        unset Y26_STAGE60_M4_TAIL || true
        if [[ $route == m8n16 ]]; then
          export Y26_STAGE54_DENSE_M8=1
        elif [[ $route == m4tail ]]; then
          unset Y26_STAGE54_DENSE_M8 || true
          export Y26_STAGE60_M4_TAIL=1
        else
          unset Y26_STAGE54_DENSE_M8 || true
        fi
        "$bin/stage60_shape_matrix" "$package" "$manifest" 5 100 \
          >"$ram/r${resolution}_${route}.tsv" 2>"$ram/r${resolution}_${route}.stderr"
        cp "$ram/r${resolution}_${route}.tsv" "$output/r${resolution}_${route}.tsv"
        cp "$ram/r${resolution}_${route}.stderr" "$output/r${resolution}_${route}.stderr"
        for event in instructions backend_stalls frontend_stalls l1d_read_access l1d_read_miss; do
          export Y26_STAGE55_PERF_GROUP="cycles,$event"
          "$bin/stage60_shape_matrix" "$package" "$manifest" 2 10 \
            >"$ram/r${resolution}_${route}_${event}.tsv" \
            2>"$ram/r${resolution}_${route}_${event}.counters"
          cp "$ram/r${resolution}_${route}_${event}.tsv" \
            "$output/r${resolution}_${route}_${event}.tsv"
          cp "$ram/r${resolution}_${route}_${event}.counters" \
            "$output/r${resolution}_${route}_${event}.counters"
        done
        unset Y26_STAGE55_PERF_GROUP
      done
      unset Y26_STAGE60_M4_TAIL || true
    done
    ;;
  *)
    echo "invalid mode: $mode" >&2
    exit 2
    ;;
esac
