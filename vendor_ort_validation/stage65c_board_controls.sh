#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_BOARD_ROOT:?STAGE65C_BOARD_ROOT is required}"
root=$STAGE65C_BOARD_ROOT
runtime="$root/runtime/lib"
runner="$root/bin/vendor_single_model_runner"
ulimit -c 0
cd "$root"

snapshot() {
  local output=$1
  {
    printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname\t%s\n' "$(hostname)"
    printf 'boot_id\t%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
    printf 'kernel\t%s\n' "$(uname -srvmo)"
    printf 'os_release\t%s\n' "$(tr '\n' ';' </etc/os-release)"
    printf 'device_model\t%s\n' "$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    printf 'device_serial\t%s\n' "$(tr -d '\0' </proc/device-tree/serial-number 2>/dev/null || true)"
    printf 'allowed_cpu_list\t%s\n' "$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
    printf 'memory\t%s\n' "$(awk '/MemTotal|MemAvailable/{printf "%s=%s%s;",$1,$2,$3}' /proc/meminfo)"
    printf 'data_mount\t%s\n' "$(findmnt -n -o TARGET,SOURCE,FSTYPE,OPTIONS --target /data)"
    printf 'root_mount\t%s\n' "$(findmnt -n -o TARGET,SOURCE,FSTYPE,OPTIONS --target /)"
    printf 'data_free_bytes\t%s\n' "$(df -B1 --output=avail /data | tail -1 | tr -d ' ')"
    for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      printf 'governor\t%s\t%s\n' "$file" "$(cat "$file")"
    done
    for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
      printf 'frequency_khz\t%s\t%s\n' "$file" "$(cat "$file")"
    done
    for file in /sys/class/thermal/thermal_zone*/temp; do
      printf 'temperature_millic\t%s\t%s\n' "$file" "$(cat "$file")"
    done
  } >"$output"
}

snapshot "$root/state/system_state_before.raw.tsv"
{
  printf 'file\tsha256\tbuild_id\n'
  for file in \
    "$runtime/libonnxruntime.so.1.24.2+spacemit.a1" \
    "$runtime/libspacemit_ep.so.2.0.6" \
    "$runner" \
    "$root/bin/stage64_two_stage_runner" \
    "$root/bin/stage64_two_stage_coco"; do
    build_id=$(readelf -n "$file" 2>/dev/null | awk '/Build ID:/{print $3; exit}')
    printf '%s\t%s\t%s\n' "$file" "$(sha256sum "$file" | awk '{print $1}')" "${build_id:-none}"
  done
} >"$root/state/loaded_library_identity.raw.tsv"
LD_LIBRARY_PATH="$runtime:$root/opencv/lib" ldd "$root/bin/stage64_two_stage_coco" \
  >"$root/state/two_stage_coco.ldd.txt"

BOARD_ROOT="$root" \
STAGE63_ROOT="$root/vendor-control" \
  "$root/bin/stage64_run_tiny_board.sh" \
  >"$root/tiny-controls/driver.stdout" \
  2>"$root/tiny-controls/driver.stderr"

plugin_case() {
  local label=$1
  local model=$2
  local input=$3
  local oracle=$4
  local plugin=$5
  local runs=$6
  local output="$root/plugin/${label}.output.bin"
  local log="$root/plugin/${label}.log"
  local profile="$root/plugin/${label}.profile"
  local rc=0
  mkdir -p "$profile" "$root/plugin/tmp" "$root/plugin/cache"
  set +e
  env \
    LD_LIBRARY_PATH="$runtime" \
    TMPDIR="$root/plugin/tmp" \
    XDG_CACHE_HOME="$root/plugin/cache" \
    timeout --signal=TERM --kill-after=5s 120s \
    taskset -c 0-3 "$runner" \
      --provider spacemit \
      --model "$model" \
      --input "$input" \
      --output "$output" \
      --opt-level disable \
      --execution-mode sequential \
      --intra-threads 1 \
      --inter-threads 1 \
      --thread-spinning 0 \
      --log-severity 1 \
      --log-verbosity 1 \
      --warmup 0 \
      --runs "$runs" \
      --repeats 1 \
      --profile-prefix "$profile/ort-profile" \
      --custom-op-library "$plugin" \
      --provider-option "SPACEMIT_EP_PLUGIN_LIB=$plugin" \
      >"$log" 2>&1
  rc=$?
  set -e
  local exact=0
  [[ $rc -eq 0 && -f $output ]] && cmp -s "$output" "$oracle" && exact=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$rc" "$exact" \
    "$(sha256sum "$output" 2>/dev/null | awk '{print $1}')" \
    "$(sha256sum "$oracle" | awk '{print $1}')" \
    "$log" >>"$root/plugin/plugin_nonregression.raw.tsv"
}

printf 'arm\texit_code\texact\toutput_sha256\toracle_sha256\tlog\n' \
  >"$root/plugin/plugin_nonregression.raw.tsv"
plugin_case \
  official_qgelu \
  "$root/plugin/official-test-files/ln_qgelu.onnx" \
  "$root/plugin/plugin_input_1x128x256_f32.bin" \
  "$root/plugin/outputs/official_track1_ep.bin" \
  "$root/plugin/libcustom_plugin.so" \
  1
plugin_case \
  independent_u8_xor \
  "$root/plugin/stage46_u8_xor.onnx" \
  "$root/plugin/stage46_u8_xor_input.bin" \
  "$root/plugin/stage46_u8_xor_expected.bin" \
  "$root/plugin/libstage46_u8_xor_plugin.so" \
  10

snapshot "$root/state/system_state_after_controls.raw.tsv"
printf 'stage65c_board_controls status=pass\n'
