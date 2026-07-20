#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: stage60_board_camera.sh compare STAGE_ROOT RES_A SHA_A RES_B SHA_B
       stage60_board_camera.sh soak STAGE_ROOT RESOLUTION MANIFEST_SHA256

Runs the Stage60 research-only static-resolution demo on the matched
640x480 MJPG GUI surface. The reversible Stage59 camera IRQ profile is applied
for each arm and temperature/frequency samples are buffered under /dev/shm.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
mode=$1
shift

case $mode in
  compare)
    [[ $# -eq 5 ]] || { usage >&2; exit 2; }
    stage_root=$1
    resolution_a=$2
    manifest_a=$3
    resolution_b=$4
    manifest_b=$5
    duration=${Y26_STAGE60_CAMERA_COMPARE_SECONDS:-210}
    ;;
  soak)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    stage_root=$1
    resolution_a=$2
    manifest_a=$3
    duration=${Y26_STAGE60_CAMERA_SOAK_SECONDS:-1830}
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

demo=$stage_root/bin/y26_k1x_demo
labels=$stage_root/labels/coco80.txt
profile=${Y26_STAGE60_CAMERA_PROFILE:-$stage_root/tools/camera-system-profile.sh}
output=$stage_root/camera/$mode
ram=/dev/shm/y26-stage60-camera-$mode-$$
mkdir -p "$output" "$ram" "$stage_root/camera/screenshots"

for required in "$demo" "$labels" "$profile"; do
  [[ -f $required ]] || { echo "missing Stage60 camera dependency: $required" >&2; exit 1; }
done

export DISPLAY=${DISPLAY:-:0}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export Y26_CAMERA_PROFILE_STATE_ROOT=$stage_root/camera/profile-state
export LD_LIBRARY_PATH=$stage_root/lib:$stage_root/lib/opencv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

sampler_pid=
stop_file=
cleanup() {
  if [[ -n $stop_file ]]; then touch "$stop_file"; fi
  if [[ -n $sampler_pid ]]; then wait "$sampler_pid" 2>/dev/null || true; fi
  "$profile" restore-stale >/dev/null 2>&1 || true
  rm -rf "$ram"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

run_arm() {
  local resolution=$1 manifest=$2 arm=$3
  local prefix=$ram/$arm
  local final_prefix=$output/$arm
  stop_file=$prefix.stop
  rm -f "$stop_file"
  printf 'timestamp_utc\tresolution\tarm\tmean_thermal_c\tmean_cpu0_4_khz\n' >"$prefix.system.tsv"
  (
    while [[ ! -e $stop_file ]]; do
      local_temperature=$(awk '{sum += $1; count += 1} END {if (count) printf "%.3f", sum/count/1000}' \
        /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true)
      local_frequency=$(awk '{sum += $1; count += 1} END {if (count) printf "%.0f", sum/count}' \
        /sys/devices/system/cpu/cpu[0-4]/cpufreq/scaling_cur_freq 2>/dev/null || true)
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$resolution" "$arm" "$local_temperature" "$local_frequency" >>"$prefix.system.tsv"
      sleep 5
    done
  ) &
  sampler_pid=$!

  set +e
  "$profile" run -- "$demo" \
    --package "$stage_root/packages/r$resolution" \
    --labels "$labels" \
    --expected-manifest-sha256 "$manifest" \
    --source camera:auto \
    --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG \
    --profile low-latency --flow latest-frame --capture-cpu 5 \
    --duration "$duration" --warmup-frames 30 --opencv-threads 1 \
    --reconnect-attempts 3 --reuse-buffers 1 \
    --metrics-tsv "$prefix.metrics.tsv" \
    --detections-tsv "$prefix.detections.tsv" \
    --log-file "$prefix.application.log" \
    --screenshot-dir "$stage_root/camera/screenshots" \
    >"$prefix.stdout.log" 2>"$prefix.stderr.log"
  status=$?
  set -e

  touch "$stop_file"
  wait "$sampler_pid" || true
  sampler_pid=
  stop_file=
  printf '%s\n' "$status" >"$prefix.exit-status.txt"
  for suffix in metrics.tsv detections.tsv application.log stdout.log stderr.log \
                system.tsv exit-status.txt; do
    if [[ -f $prefix.$suffix ]]; then
      cp "$prefix.$suffix" "$final_prefix.$suffix"
    fi
  done
  if [[ $status -ne 0 ]]; then
    echo "camera arm failed: $arm" >&2
    return "$status"
  fi
  [[ -s $final_prefix.metrics.tsv && -s $final_prefix.detections.tsv ]] || {
    echo "camera arm did not produce complete metrics/detections: $arm" >&2
    return 1
  }
  "$profile" status >"$final_prefix.profile-after.txt"
}

if [[ $mode == compare ]]; then
  declare -A count=([$resolution_a]=0 [$resolution_b]=0)
  for resolution in "$resolution_a" "$resolution_b" "$resolution_b" \
                    "$resolution_a" "$resolution_a" "$resolution_b"; do
    count[$resolution]=$((count[$resolution] + 1))
    if [[ $resolution == "$resolution_a" ]]; then manifest=$manifest_a; else manifest=$manifest_b; fi
    run_arm "$resolution" "$manifest" "r${resolution}-run${count[$resolution]}"
  done
else
  run_arm "$resolution_a" "$manifest_a" "r${resolution_a}-soak"
fi
