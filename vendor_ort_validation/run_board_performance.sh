#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 canonical|canonical-blocks|stability|thread-scout" >&2
  exit 2
fi

mode=$1
: "${STAGE63_BOARD_ROOT:?set STAGE63_BOARD_ROOT}"

root=$STAGE63_BOARD_ROOT
out_root="$root/performance/$mode"
mkdir -p "$out_root/raw" "$out_root/outputs" "$out_root/state"
status_tsv="$out_root/status.tsv"
printf 'arm\truntime\tprovider\tsurface\twarmup\ttotal_inferences\truns_per_repeat\trepeats\texit_code\tsignal\ttimed_out\toutput_sha256\tstatus\n' \
  >"$status_tsv"

snapshot_state() {
  local output=$1
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'affinity='
    taskset -pc $$ 2>&1
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
      [[ -r "$path" ]] && printf '%s=%s\n' "$path" "$(<"$path")"
    done
    for path in /sys/class/thermal/thermal_zone*/temp; do
      [[ -r "$path" ]] && printf '%s=%s\n' "$path" "$(<"$path")"
    done
  } >"$output"
}

run_arm() {
  local runtime=$1
  local provider=$2
  local surface=$3
  local warmup=$4
  local samples=$5
  local threads=$6
  local label=${7:-"${runtime}_${provider}_${surface}_t${threads}"}
  local runs_per_repeat=${8:-1}
  local repeats=${9:-$samples}
  local runner="$root/runners/runner_${runtime}"
  local runtime_lib="$root/runtimes/${runtime}/lib"
  local model
  case "$surface" in
    fp32) model="$root/models/yolo26n_640_e2e_fp32.onnx" ;;
    fp16) model="$root/models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx" ;;
    int8) model="$root/models/manual_e2e_rep_conv_matmul_qdq.onnx" ;;
    *) echo "unsupported surface: $surface" >&2; return 2 ;;
  esac

  local log="$out_root/raw/${label}.log"
  local time_log="$out_root/raw/${label}.time.txt"
  local output="$out_root/outputs/${label}.bin"
  local command_log="$out_root/raw/${label}.command.txt"
  rm -f "$output"
  printf '%q ' taskset -c 0-3 env "LD_LIBRARY_PATH=$runtime_lib" \
    "SPACEMIT_EP_DUMP_SUBGRAPHS=0" "$runner" --provider "$provider" \
    --model "$model" --input "$root/fixtures/preprocessed/images_F0_f32.bin" \
    --output "$output" --opt-level all --execution-mode sequential \
    --intra-threads "$threads" --inter-threads 1 --thread-spinning 0 \
    --warmup "$warmup" --runs "$runs_per_repeat" --repeats "$repeats" >"$command_log"
  printf '\n' >>"$command_log"

  snapshot_state "$out_root/state/${label}.before.txt"
  set +e
  taskset -c 0-3 env LD_LIBRARY_PATH="$runtime_lib" SPACEMIT_EP_DUMP_SUBGRAPHS=0 \
    timeout --signal=TERM --kill-after=5s 3600s \
    /usr/bin/time -v -o "$time_log" "$runner" \
      --provider "$provider" --model "$model" \
      --input "$root/fixtures/preprocessed/images_F0_f32.bin" --output "$output" \
      --opt-level all --execution-mode sequential --intra-threads "$threads" \
      --inter-threads 1 --thread-spinning 0 --warmup "$warmup" \
      --runs "$runs_per_repeat" --repeats "$repeats" >"$log" 2>&1
  local rc=$?
  set -e
  snapshot_state "$out_root/state/${label}.after.txt"

  local signal=0
  local timed_out=0
  local output_sha=missing
  local status=fail
  if ((rc >= 128 && rc < 192)); then signal=$((rc - 128)); fi
  if ((rc == 124 || rc == 137 || rc == 143)); then timed_out=1; fi
  if ((rc == 0)) && [[ -f "$output" ]]; then
    output_sha=$(sha256sum "$output" | awk '{print $1}')
    status=pass
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$runtime" "$provider" "$surface" "$warmup" "$samples" \
    "$runs_per_repeat" "$repeats" "$rc" "$signal" "$timed_out" "$output_sha" \
    "$status" >>"$status_tsv"
  printf 'completed arm=%s status=%s exit=%s\n' "$label" "$status" "$rc"
}

case "$mode" in
  canonical)
    run_arm b120 cpu int8 10 500 4
    run_arm rt204 cpu int8 10 500 4
    run_arm rt205 cpu int8 10 500 4
    run_arm rt206 cpu int8 10 500 4
    run_arm rt204 spacemit fp32 10 500 4
    run_arm rt205 spacemit fp32 10 500 4
    run_arm rt206 spacemit fp32 10 500 4
    run_arm rt204 spacemit fp16 10 500 4
    run_arm rt205 spacemit fp16 10 500 4
    run_arm rt206 spacemit fp16 10 500 4
    run_arm rt206 cpu fp32 10 500 4
    run_arm rt206 cpu fp16 10 500 4
    ;;
  canonical-blocks)
    run_arm b120 cpu int8 10 500 4 b120_cpu_int8_t4_blocks 100 5
    run_arm rt204 cpu int8 10 500 4 rt204_cpu_int8_t4_blocks 100 5
    run_arm rt205 cpu int8 10 500 4 rt205_cpu_int8_t4_blocks 100 5
    run_arm rt206 cpu int8 10 500 4 rt206_cpu_int8_t4_blocks 100 5
    run_arm rt204 spacemit fp32 10 500 4 rt204_spacemit_fp32_t4_blocks 100 5
    run_arm rt205 spacemit fp32 10 500 4 rt205_spacemit_fp32_t4_blocks 100 5
    run_arm rt206 spacemit fp32 10 500 4 rt206_spacemit_fp32_t4_blocks 100 5
    run_arm rt204 spacemit fp16 10 500 4 rt204_spacemit_fp16_t4_blocks 100 5
    run_arm rt205 spacemit fp16 10 500 4 rt205_spacemit_fp16_t4_blocks 100 5
    run_arm rt206 spacemit fp16 10 500 4 rt206_spacemit_fp16_t4_blocks 100 5
    run_arm rt206 cpu fp32 10 500 4 rt206_cpu_fp32_t4_blocks 100 5
    run_arm rt206 cpu fp16 10 500 4 rt206_cpu_fp16_t4_blocks 100 5
    ;;
  stability)
    run_arm rt206 cpu int8 10 1000 4 rt206_cpu_int8_stability
    run_arm rt206 cpu fp32 10 1000 4 rt206_cpu_fp32_stability
    run_arm rt206 cpu fp16 10 1000 4 rt206_cpu_fp16_stability
    run_arm rt206 spacemit fp32 10 1000 4 rt206_spacemit_fp32_stability
    run_arm rt206 spacemit fp16 10 1000 4 rt206_spacemit_fp16_stability
    ;;
  thread-scout)
    for threads in 1 2 4; do
      run_arm rt206 cpu int8 3 20 "$threads"
    done
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
