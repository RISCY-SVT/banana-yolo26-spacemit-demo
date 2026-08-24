#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
root=$STAGE65D_BOARD_ROOT
runner="$root/bin/stage64_two_stage_runner"
runtime="$root/runtime/lib"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
out="$root/performance/abba"
ulimit -c 0
mkdir -p "$out"
[[ ! -e $out/status.raw.tsv ]] || { printf 'performance output exists: %s\n' "$out" >&2; exit 2; }
printf 'phase\tblock\tslot\tmodel\texit_code\toutput_sha256\tsamples_sha256\n' >"$out/status.raw.tsv"
printf 'phase\tblock\tslot\tmodel\tposition\n' >"$out/schedule.raw.tsv"

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

run_slot() {
  local phase=$1 block=$2 slot=$3 model=$4
  local label="${phase}-b${block}-s${slot}-${model}"
  local slot_dir="$out/$label"
  local inference
  inference=$(inference_for "$model")
  mkdir -p "$slot_dir/tmp" "$slot_dir/cache"
  snapshot "$slot_dir/state-before.tsv"
  rc=0
  set +e
  env LD_LIBRARY_PATH="$runtime" TMPDIR="$slot_dir/tmp" XDG_CACHE_HOME="$slot_dir/cache" \
    timeout --signal=TERM --kill-after=10s 600s \
    taskset -c 0-3 "$runner" \
      --provider spacemit \
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
      >"$slot_dir/run.log" 2>&1
  rc=$?
  set -e
  snapshot "$slot_dir/state-after.tsv"
  output_hash=missing
  samples_hash=missing
  [[ -f $slot_dir/output.bin ]] && output_hash=$(sha256sum "$slot_dir/output.bin" | awk '{print $1}')
  [[ -f $slot_dir/samples.tsv ]] && samples_hash=$(sha256sum "$slot_dir/samples.tsv" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$phase" "$block" "$slot" "$model" "$rc" "$output_hash" "$samples_hash" \
    >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || exit "$rc"
}

# Two bounded B2/B2 blocks establish process/session and ordering noise.
for block in 0 1; do
  for slot in 0 1 2 3; do
    printf 'noise\t%s\t%s\tB2\t%s\n' "$block" "$slot" "$slot" >>"$out/schedule.raw.tsv"
    run_slot noise "$block" "$slot" B2
  done
done

for block in 0 1 2 3 4; do
  if ((block % 2 == 0)); then
    order=(B2 C2 C2 B2)
  else
    order=(C2 B2 B2 C2)
  fi
  for slot in 0 1 2 3; do
    model=${order[$slot]}
    printf 'abba\t%s\t%s\t%s\t%s\n' "$block" "$slot" "$model" "$slot" >>"$out/schedule.raw.tsv"
    run_slot abba "$block" "$slot" "$model"
  done
done

printf 'stage65d_board_performance status=pass\n'
