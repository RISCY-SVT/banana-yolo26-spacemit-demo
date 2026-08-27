#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65E_BOARD_ROOT:?STAGE65E_BOARD_ROOT is required}"
root=$STAGE65E_BOARD_ROOT
runner="$root/bin/stage64_two_stage_runner"
runtime="$root/runtime/lib"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
short_out="$root/stability/stage65e-short-1k"
b2_out="$root/stability/stage65e-b2-10k"
c2_out="$root/stability/stage65e-c2-10k"
ulimit -c 0

for out in "$short_out" "$b2_out" "$c2_out"; do
  [[ ! -e $out ]] || { printf 'stability output exists: %s\n' "$out" >&2; exit 2; }
  mkdir -p "$out"
  printf 'segment\tposition\tmodel\truns\texit_code\toutput_sha256\tsamples_sha256\tresource_sha256\n' >"$out/status.raw.tsv"
  printf 'segment\tposition\tmodel\n' >"$out/schedule.raw.tsv"
done

declare -A accepted_output_hash

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

snapshot() {
  local file=$1
  {
    printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      [[ -r $path ]] || continue
      printf 'governor\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
      [[ -r $path ]] || continue
      printf 'frequency_khz\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/class/thermal/thermal_zone*/temp; do
      [[ -r $path ]] || continue
      printf 'temperature_millic\t%s\t%s\n' "$path" "$(cat "$path")"
    done
  } >"$file"
  [[ -s $file ]] || { printf 'empty state snapshot: %s\n' "$file" >&2; return 3; }
}

state_gate() {
  local file=$1
  awk -F '\t' '
    $1 == "governor" { governors += 1; if ($3 != "performance") bad = 1 }
    $1 == "frequency_khz" {
      frequencies += 1
      value = $3 + 0
      if (minimum == 0 || value < minimum) minimum = value
      if (value > maximum) maximum = value
    }
    $1 == "temperature_millic" {
      temperatures += 1
      if (($3 + 0) > 85000) bad = 1
    }
    END {
      if (governors == 0 || frequencies == 0 || temperatures == 0 || maximum == 0) bad = 1
      if (minimum < 0.95 * maximum) bad = 1
      exit bad ? 1 : 0
    }
  ' "$file"
}

resource_gate() {
  local file=$1
  awk -F '\t' '
    NR == 1 { next }
    ($3 + 0) <= 0 || ($5 + 0) <= 0 || ($6 + 0) <= 0 { next }
    ($1 + 0) < 30 { next }
    {
      count += 1
      rss[count] = $3 + 0
      fd[count] = $5 + 0
      threads[count] = $6 + 0
    }
    END {
      usable = count - 1
      if (usable < 1) exit 1
      first_rss = rss[1]
      last_rss = rss[usable]
      min_fd = max_fd = fd[1]
      min_threads = max_threads = threads[1]
      for (index = 1; index <= usable; index += 1) {
        if (fd[index] < min_fd) min_fd = fd[index]
        if (fd[index] > max_fd) max_fd = fd[index]
        if (threads[index] < min_threads) min_threads = threads[index]
        if (threads[index] > max_threads) max_threads = threads[index]
      }
      bad = last_rss > first_rss + 16384 || max_fd - min_fd > 2 || max_threads - min_threads > 2
      exit bad ? 1 : 0
    }
  ' "$file"
}

output_semantic_gate() {
  local file=$1
  python3 - "$file" <<'PY'
import math
import struct
import sys

payload = open(sys.argv[1], "rb").read()
if len(payload) != 1800 * 4:
    raise SystemExit(1)
values = struct.unpack("<1800f", payload)
if not all(math.isfinite(value) for value in values):
    raise SystemExit(1)
scores = values[4::6]
if len(set(scores)) < 2:
    raise SystemExit(1)
PY
}

run_segment() {
  local out=$1 segment=$2 position=$3 model=$4 runs=$5
  local segment_dir="$out/segment-${segment}-${position}-${model}"
  local inference pid rc=0 output_hash=missing samples_hash=missing resource_hash
  inference=$(inference_for "$model")
  mkdir -p "$segment_dir/tmp" "$segment_dir/cache"
  snapshot "$segment_dir/state-before.tsv"
  state_gate "$segment_dir/state-before.tsv"
  printf 'sample\ttimestamp_utc\trss_kib\tpeak_rss_kib\tfds\tthreads\tvoluntary_ctxt\tnonvoluntary_ctxt\n' >"$segment_dir/resource.tsv"

  env LD_LIBRARY_PATH="$runtime" TMPDIR="$segment_dir/tmp" XDG_CACHE_HOME="$segment_dir/cache" \
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
      --runs "$runs" \
      --repeats 1 \
      >"$segment_dir/run.log" 2>&1 &
  pid=$!
  local sample=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -r /proc/$pid/status ]]; then
      local rss peak fds threads voluntary nonvoluntary
      rss=$(awk '$1 == "VmRSS:" {print $2}' "/proc/$pid/status" 2>/dev/null || true)
      peak=$(awk '$1 == "VmHWM:" {print $2}' "/proc/$pid/status" 2>/dev/null || true)
      threads=$(awk '$1 == "Threads:" {print $2}' "/proc/$pid/status" 2>/dev/null || true)
      voluntary=$(awk '$1 == "voluntary_ctxt_switches:" {print $2}' "/proc/$pid/status" 2>/dev/null || true)
      nonvoluntary=$(awk '$1 == "nonvoluntary_ctxt_switches:" {print $2}' "/proc/$pid/status" 2>/dev/null || true)
      fds=$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l || true)
      if [[ -z $rss || -z $peak || -z $fds || -z $threads || $fds -le 0 ]]; then
        sleep 1
        continue
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sample" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rss:-0}" "${peak:-0}" \
        "$fds" "${threads:-0}" "${voluntary:-0}" "${nonvoluntary:-0}" \
        >>"$segment_dir/resource.tsv"
      sample=$((sample + 1))
    fi
    sleep 1
  done

  wait "$pid" || rc=$?
  snapshot "$segment_dir/state-after.tsv"
  if ! state_gate "$segment_dir/state-after.tsv"; then
    printf 'thermal/frequency state outside accepted bounds: %s\n' "$segment_dir" >&2
    rc=3
  fi
  if ! resource_gate "$segment_dir/resource.tsv"; then
    printf 'resource state outside accepted bounds: %s\n' "$segment_dir" >&2
    rc=3
  fi
  if [[ ! -f $segment_dir/output.bin ]] || ! output_semantic_gate "$segment_dir/output.bin"; then
    printf 'output finite/non-collapse contract failed: %s\n' "$segment_dir" >&2
    rc=3
  fi
  resource_hash=$(sha256sum "$segment_dir/resource.tsv" | awk '{print $1}')
  [[ -f $segment_dir/output.bin ]] && output_hash=$(sha256sum "$segment_dir/output.bin" | awk '{print $1}')
  [[ -f $segment_dir/samples.tsv ]] && samples_hash=$(sha256sum "$segment_dir/samples.tsv" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$segment" "$position" "$model" "$runs" "$rc" "$output_hash" "$samples_hash" \
    "$resource_hash" >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || return "$rc"
  if [[ -n ${accepted_output_hash[$model]:-} && ${accepted_output_hash[$model]} != "$output_hash" ]]; then
    printf 'output hash changed for %s: %s != %s\n' "$model" "$output_hash" "${accepted_output_hash[$model]}" >&2
    return 3
  fi
  accepted_output_hash[$model]=$output_hash
}

short_order=(B2 C2 C2 B2)
for position in 0 1 2 3; do
  model=${short_order[$position]}
  printf '0\t%s\t%s\n' "$position" "$model" >>"$short_out/schedule.raw.tsv"
  run_segment "$short_out" 0 "$position" "$model" 1000
done

for segment in $(seq 0 9); do
  printf '%s\t0\tB2\n' "$segment" >>"$b2_out/schedule.raw.tsv"
  run_segment "$b2_out" "$segment" 0 B2 1000
done

for segment in $(seq 0 9); do
  printf '%s\t0\tC2\n' "$segment" >>"$c2_out/schedule.raw.tsv"
  run_segment "$c2_out" "$segment" 0 C2 1000
done

printf 'stage65e_board_soak status=pass\n'
