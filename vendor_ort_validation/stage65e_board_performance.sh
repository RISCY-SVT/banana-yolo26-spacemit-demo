#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65E_BOARD_ROOT:?STAGE65E_BOARD_ROOT is required}"
root=$STAGE65E_BOARD_ROOT
runner="$root/bin/stage64_two_stage_runner"
runtime="$root/runtime/lib"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
out="$root/performance/stage65e-passport"
ulimit -c 0

[[ ! -e $out ]] || { printf 'performance output exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$out"
printf 'phase\tblock\tslot\tmodel\tprovider\texit_code\toutput_sha256\tsamples_sha256\tresource_sha256\n' >"$out/status.raw.tsv"
printf 'phase\tblock\tslot\tmodel\tprovider\tposition\n' >"$out/schedule.raw.tsv"

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

run_slot() {
  local phase=$1 block=$2 slot=$3 model=$4 provider=$5
  local label="${phase}-b${block}-s${slot}-${model}-${provider}"
  local slot_dir="$out/$label"
  local inference pid rc=0
  inference=$(inference_for "$model")
  mkdir -p "$slot_dir/tmp" "$slot_dir/cache"
  snapshot "$slot_dir/state-before.tsv"
  printf 'sample\ttimestamp_utc\trss_kib\tpeak_rss_kib\tfds\tthreads\tvoluntary_ctxt\tnonvoluntary_ctxt\n' >"$slot_dir/resource.tsv"

  env LD_LIBRARY_PATH="$runtime" TMPDIR="$slot_dir/tmp" XDG_CACHE_HOME="$slot_dir/cache" \
    taskset -c 0-3 "$runner" \
      --provider "$provider" \
      --inference-model "$inference" \
      --tail-model "$tail_model" \
      --input "$root/fixtures/fixed/images_F0_f32.bin" \
      --output "$slot_dir/output.bin" \
      --samples-output "$slot_dir/samples.tsv" \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup 10 \
      --runs 100 \
      --repeats 1 \
      >"$slot_dir/run.log" 2>&1 &
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
      if [[ -z $rss || -z $threads ]]; then
        sleep 0.25
        continue
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sample" "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "${rss:-0}" "${peak:-0}" \
        "$fds" "${threads:-0}" "${voluntary:-0}" "${nonvoluntary:-0}" \
        >>"$slot_dir/resource.tsv"
      sample=$((sample + 1))
    fi
    sleep 0.25
  done

  wait "$pid" || rc=$?
  snapshot "$slot_dir/state-after.tsv"
  local output_hash=missing samples_hash=missing resource_hash
  resource_hash=$(sha256sum "$slot_dir/resource.tsv" | awk '{print $1}')
  [[ -f $slot_dir/output.bin ]] && output_hash=$(sha256sum "$slot_dir/output.bin" | awk '{print $1}')
  [[ -f $slot_dir/samples.tsv ]] && samples_hash=$(sha256sum "$slot_dir/samples.tsv" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$phase" "$block" "$slot" "$model" "$provider" "$rc" \
    "$output_hash" "$samples_hash" "$resource_hash" >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -s $slot_dir/samples.tsv && -s $slot_dir/output.bin ]] || return 3
}

run_noise() {
  local model=$1 phase=$2
  for block in $(seq 0 7); do
    for slot in 0 1 2 3; do
      printf '%s\t%s\t%s\t%s\tspacemit\t%s\n' "$phase" "$block" "$slot" "$model" "$slot" >>"$out/schedule.raw.tsv"
      run_slot "$phase" "$block" "$slot" "$model" spacemit
    done
  done
}

run_noise B2 noise_b2
run_noise C2 noise_c2

for block in $(seq 0 11); do
  if ((block % 2 == 0)); then
    order=(B2 C2 C2 B2)
  else
    order=(C2 B2 B2 C2)
  fi
  for slot in 0 1 2 3; do
    model=${order[$slot]}
    printf 'abba\t%s\t%s\t%s\tspacemit\t%s\n' "$block" "$slot" "$model" "$slot" >>"$out/schedule.raw.tsv"
    run_slot abba "$block" "$slot" "$model" spacemit
  done
done

for block in 0 1 2 3; do
  if ((block % 2 == 0)); then
    order=(B2 C2)
  else
    order=(C2 B2)
  fi
  for slot in 0 1; do
    model=${order[$slot]}
    printf 'cpu\t%s\t%s\t%s\tcpu\t%s\n' "$block" "$slot" "$model" "$slot" >>"$out/schedule.raw.tsv"
    run_slot cpu "$block" "$slot" "$model" cpu
  done
done

printf 'stage65e_board_performance status=pass\n'
