#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_BOARD_ROOT:?STAGE65C_BOARD_ROOT is required}"
root=$STAGE65C_BOARD_ROOT
runner="$root/bin/stage64_two_stage_runner"
runtime="$root/runtime/lib"
tail_model="$root/models/stage65b_r1_a1.postprocess.onnx"
out="$root/stability/soak10k"
ulimit -c 0
mkdir -p "$out"
[[ ! -e $out/status.raw.tsv ]] || {
  printf 'stability output exists: %s\n' "$out" >&2
  exit 2
}
printf 'segment\tposition\tmodel\truns\texit_code\toutput_sha256\tsamples_sha256\tresource_sha256\n' \
  >"$out/status.raw.tsv"
printf 'segment\tposition\tmodel\n' >"$out/schedule.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    A1) printf '%s' "$root/models/stage65b_r1_a1.inference.onnx" ;;
    *) return 2 ;;
  esac
}

snapshot() {
  local file=$1
  {
    printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      printf 'governor\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
      printf 'frequency_khz\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/class/thermal/thermal_zone*/temp; do
      printf 'temperature_millic\t%s\t%s\n' "$path" "$(cat "$path")"
    done
  } >"$file"
}

run_segment() {
  local segment=$1 position=$2 model=$3
  local segment_dir="$out/segment-${segment}-${position}-${model}"
  local inference pid watchdog rc output_hash samples_hash resource_hash
  inference=$(inference_for "$model")
  mkdir -p "$segment_dir/tmp" "$segment_dir/cache"
  snapshot "$segment_dir/state-before.tsv"
  printf 'sample\ttimestamp_utc\trss_kib\tpeak_rss_kib\tfds\tthreads\tvoluntary_ctxt\tnonvoluntary_ctxt\n' \
    >"$segment_dir/resource.tsv"

  env LD_LIBRARY_PATH="$runtime" TMPDIR="$segment_dir/tmp" \
    XDG_CACHE_HOME="$segment_dir/cache" \
    taskset -c 0-3 "$runner" \
      --provider spacemit \
      --inference-model "$inference" \
      --tail-model "$tail_model" \
      --input "$root/fixtures/fixed/images_F0_f32.bin" \
      --output "$segment_dir/output.bin" \
      --samples-output "$segment_dir/samples.tsv" \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup 10 \
      --runs 1000 \
      --repeats 1 \
      >"$segment_dir/run.log" 2>&1 &
  pid=$!
  (
    sleep 900
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 10
      kill -KILL "$pid" 2>/dev/null || true
    fi
  ) &
  watchdog=$!

  sample=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -r /proc/$pid/status ]]; then
      rss=$(awk '$1 == "VmRSS:" {print $2}' "/proc/$pid/status")
      peak=$(awk '$1 == "VmHWM:" {print $2}' "/proc/$pid/status")
      threads=$(awk '$1 == "Threads:" {print $2}' "/proc/$pid/status")
      voluntary=$(awk '$1 == "voluntary_ctxt_switches:" {print $2}' "/proc/$pid/status")
      nonvoluntary=$(awk '$1 == "nonvoluntary_ctxt_switches:" {print $2}' "/proc/$pid/status")
      fds=$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sample" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rss:-0}" \
        "${peak:-0}" "$fds" "${threads:-0}" "${voluntary:-0}" \
        "${nonvoluntary:-0}" >>"$segment_dir/resource.tsv"
      sample=$((sample + 1))
    fi
    sleep 1
  done

  rc=0
  wait "$pid" || rc=$?
  kill "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  snapshot "$segment_dir/state-after.tsv"

  output_hash=missing
  samples_hash=missing
  resource_hash=$(sha256sum "$segment_dir/resource.tsv" | awk '{print $1}')
  [[ -f $segment_dir/output.bin ]] &&
    output_hash=$(sha256sum "$segment_dir/output.bin" | awk '{print $1}')
  [[ -f $segment_dir/samples.tsv ]] &&
    samples_hash=$(sha256sum "$segment_dir/samples.tsv" | awk '{print $1}')
  printf '%s\t%s\t%s\t1000\t%s\t%s\t%s\t%s\n' \
    "$segment" "$position" "$model" "$rc" "$output_hash" "$samples_hash" \
    "$resource_hash" >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || return "$rc"
}

for segment in 0 1 2 3 4 5 6 7 8 9; do
  if ((segment % 2 == 0)); then
    order=(B2 A1)
  else
    order=(A1 B2)
  fi
  for position in 0 1; do
    model=${order[$position]}
    printf '%s\t%s\t%s\n' "$segment" "$position" "$model" \
      >>"$out/schedule.raw.tsv"
    run_segment "$segment" "$position" "$model"
  done
done

printf 'stage65c_board_soak status=pass\n'
