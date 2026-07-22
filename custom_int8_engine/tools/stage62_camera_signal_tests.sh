#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: stage62_camera_signal_tests.sh STAGE_ROOT R640_MANIFEST_SHA256

Runs bounded camera signal and injected-failure lifecycle tests. The script
expects the Stage62 board deployment layout and writes under camera/signal-failure.
EOF
}

[[ $# -eq 2 ]] || {
  usage >&2
  exit 2
}

stage_root=$1
manifest_sha256=$2
output=$stage_root/camera/signal-failure
demo=$stage_root/bin/y26_k1x_demo
profile=$stage_root/tools/camera-system-profile.sh

mkdir -p "$output"
export LD_LIBRARY_PATH="$stage_root/lib:$stage_root/lib/opencv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DISPLAY=${DISPLAY:-:0}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export Y26_CAMERA_PROFILE_STATE_ROOT=$stage_root/camera/profile-state

common=(
  --package "$stage_root/packages/r640"
  --model-resolution 640
  --labels "$stage_root/labels/coco80.txt"
  --expected-manifest-sha256 "$manifest_sha256"
  --source camera:auto
  --camera-width 640
  --camera-height 480
  --camera-fps 60
  --camera-fourcc MJPG
  --profile low-latency
  --flow latest-frame
  --capture-cpu 5
  --duration 120
  --warmup-frames 5
  --opencv-threads 1
  --reconnect-attempts 3
  --reuse-buffers 1
  --headless
)

profile_is_restored() {
  [[ ! -e $stage_root/camera/profile-state/irq-state.tsv ]]
}

run_signal() {
  local signal=$1
  local name=${signal,,}
  local wrapper child status metrics detections frame stale processes

  rm -f "$output/$name".*
  set +e
  "$profile" run -- "$demo" "${common[@]}" \
    --metrics-tsv "$output/$name.metrics.tsv" \
    --detections-tsv "$output/$name.detections.tsv" \
    --log-file "$output/$name.application.log" \
    --save-frame "$output/$name.annotated.png" \
    >"$output/$name.stdout.log" 2>"$output/$name.stderr.log" &
  wrapper=$!
  set -e

  child=
  for _ in $(seq 1 60); do
    child=$(pgrep -P "$wrapper" -f y26_k1x_demo | head -1 || true)
    [[ -n $child ]] && break
    sleep 0.25
  done
  if [[ -z $child ]]; then
    echo "signal=$signal result=FAIL reason=no-child wrapper=$wrapper"
    kill -TERM "$wrapper" 2>/dev/null || true
    wait "$wrapper" || true
    return 1
  fi

  sleep 10
  kill -s "$signal" "$child"
  set +e
  wait "$wrapper"
  status=$?
  set -e
  printf '%s\n' "$status" >"$output/$name.exit-status.txt"
  "$profile" status >"$output/$name.profile-after.txt"

  metrics=0
  detections=0
  frame=0
  stale=0
  [[ -s $output/$name.metrics.tsv ]] && metrics=1
  [[ -s $output/$name.detections.tsv ]] && detections=1
  [[ -s $output/$name.annotated.png ]] && frame=1
  profile_is_restored || stale=1
  processes=$(pgrep -fc "y26_k1x_demo.*$name.metrics.tsv" || true)

  echo "signal=$signal wrapper_status=$status metrics=$metrics detections=$detections frame=$frame stale_state=$stale leftover_processes=$processes"
  [[ $status -eq 0 && $metrics -eq 1 && $detections -eq 1 && $stale -eq 0 && $processes -eq 0 ]]
}

run_capture_failure() {
  local name=capture-failure status stale=0
  rm -f "$output/$name".*
  set +e
  "$profile" run -- "$demo" \
    --package "$stage_root/packages/r640" \
    --model-resolution 640 \
    --labels "$stage_root/labels/coco80.txt" \
    --expected-manifest-sha256 "$manifest_sha256" \
    --source camera:/dev/video999 \
    --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG \
    --profile low-latency --flow latest-frame --capture-cpu 5 \
    --duration 15 --warmup-frames 1 --opencv-threads 1 \
    --reconnect-attempts 1 --reuse-buffers 1 --headless \
    --metrics-tsv "$output/$name.metrics.tsv" \
    --detections-tsv "$output/$name.detections.tsv" \
    --log-file "$output/$name.application.log" \
    >"$output/$name.stdout.log" 2>"$output/$name.stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$output/$name.exit-status.txt"
  "$profile" status >"$output/$name.profile-after.txt"
  profile_is_restored || stale=1
  echo "failure=capture-open status=$status expected_nonzero=1 stale_state=$stale"
  [[ $status -ne 0 && $stale -eq 0 ]]
}

run_recorder_failure() {
  local name=recorder-failure status stale=0 failures
  rm -f "$output/$name".*
  set +e
  "$profile" run -- "$demo" "${common[@]}" \
    --duration 20 \
    --record /proc/y26-stage62-recorder-failure.avi \
    --record-mode async \
    --metrics-tsv "$output/$name.metrics.tsv" \
    --detections-tsv "$output/$name.detections.tsv" \
    --log-file "$output/$name.application.log" \
    --save-frame "$output/$name.annotated.png" \
    >"$output/$name.stdout.log" 2>"$output/$name.stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$output/$name.exit-status.txt"
  "$profile" status >"$output/$name.profile-after.txt"
  profile_is_restored || stale=1
  failures=$(sed -n 's/.*recording_failures=\([0-9][0-9]*\).*/\1/p' \
    "$output/$name.stdout.log" | tail -1)
  failures=${failures:-0}
  echo "failure=recorder-open status=$status recording_failures=$failures metrics=$([[ -s $output/$name.metrics.tsv ]] && echo 1 || echo 0) stale_state=$stale"
  [[ $status -eq 0 && $failures -ge 1 && -s $output/$name.metrics.tsv && $stale -eq 0 ]]
}

for signal in INT TERM HUP; do
  run_signal "$signal"
done
run_capture_failure
run_recorder_failure

"$profile" status
if pgrep -af y26_k1x_demo; then
  echo "leftover demo process detected" >&2
  exit 1
fi
echo 'signal_failure_matrix=PASS'
