#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: stage61_board_camera.sh probe STAGE_ROOT DEVICE WIDTH HEIGHT FPS FOURCC
       stage61_board_camera.sh compare STAGE_ROOT WIDTH HEIGHT RES_A SHA_A RES_B SHA_B
       stage61_board_camera.sh soak STAGE_ROOT WIDTH HEIGHT RESOLUTION MANIFEST_SHA256

The compare and soak modes retain the accepted Stage59 schema-v2 latest-frame
GUI surface. The probe mode records direct V4L2 MMAP dequeue telemetry in a
separate run; it is not concurrent with OpenCV capture.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
mode=$1
shift

case $mode in
  probe)
    [[ $# -eq 6 ]] || { usage >&2; exit 2; }
    stage_root=$1
    device=$2
    width=$3
    height=$4
    fps=$5
    fourcc=$6
    duration=${Y26_STAGE61_V4L2_SECONDS:-180}
    output=$stage_root/camera/${width}x${height}/v4l2
    mkdir -p "$output"
    "$stage_root/bin/y26_v4l2_probe" "$device" "$width" "$height" "$fourcc" \
      "$fps" "$duration" "$output/dqbuf.tsv" >"$output/summary.txt"
    ;;
  compare)
    [[ $# -eq 7 ]] || { usage >&2; exit 2; }
    stage_root=$1
    width=$2
    height=$3
    resolution_a=$4
    manifest_a=$5
    resolution_b=$6
    manifest_b=$7
    duration=${Y26_STAGE61_CAMERA_COMPARE_SECONDS:-180}
    ;;
  soak)
    [[ $# -eq 5 ]] || { usage >&2; exit 2; }
    stage_root=$1
    width=$2
    height=$3
    resolution_a=$4
    manifest_a=$5
    duration=${Y26_STAGE61_CAMERA_SOAK_SECONDS:-1800}
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

[[ $mode == probe ]] && exit 0

demo=$stage_root/bin/y26_k1x_demo
labels=$stage_root/labels/coco80.txt
profile=${Y26_STAGE61_CAMERA_PROFILE:-$stage_root/tools/camera-system-profile.sh}
output=$stage_root/camera/${width}x${height}/$mode
ram=/dev/shm/y26-stage61-camera-$mode-$$
mkdir -p "$output" "$ram" "$stage_root/camera/screenshots"

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

for required in "$demo" "$labels" "$profile"; do
  [[ -f $required ]] || { echo "missing Stage61 camera dependency: $required" >&2; exit 1; }
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
      temperature=$(awk '{sum += $1; count += 1} END {if (count) printf "%.3f", sum/count/1000}' \
        /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true)
      frequency=$(awk '{sum += $1; count += 1} END {if (count) printf "%.0f", sum/count}' \
        /sys/devices/system/cpu/cpu[0-4]/cpufreq/scaling_cur_freq 2>/dev/null || true)
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$resolution" "$arm" "$temperature" "$frequency" >>"$prefix.system.tsv"
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
    --camera-width "$width" --camera-height "$height" --camera-fps 60 --camera-fourcc MJPG \
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
  sha256sum "$demo" "$stage_root/packages/r$resolution/asset_hashes.tsv" \
    "$labels" "$profile" >"$prefix.identities.txt"
  for suffix in metrics.tsv detections.tsv application.log stdout.log stderr.log \
                system.tsv exit-status.txt identities.txt; do
    [[ -f $prefix.$suffix ]] && cp "$prefix.$suffix" "$final_prefix.$suffix"
  done
  [[ $status -eq 0 ]] || { echo "camera arm failed: $arm" >&2; return "$status"; }
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
