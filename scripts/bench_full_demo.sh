#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

[[ ${1:-} != --help && ${1:-} != -h ]] || {
  echo "usage: $0 [extra y26_k1x_demo arguments]"
  exit 0
}
root=$Y26_BOARD_RELEASE_ROOT
metrics=${Y26_CAMERA_METRICS:-/data/y26-camera-full-demo.tsv}
args=("$root/bin/y26_k1x_demo" --package "$root/package" --labels "$root/labels/coco80.txt"
      --source camera:auto --camera-width "${Y26_CAMERA_WIDTH:-1280}"
      --camera-height "${Y26_CAMERA_HEIGHT:-720}" --camera-fps "${Y26_CAMERA_FPS:-60}"
      --camera-fourcc "${Y26_CAMERA_FOURCC:-MJPG}" --profile low-latency --flow latest-frame
      --headless --warmup-frames 30 --max-frames 1000 --metrics-tsv "$metrics" "$@")
echo "surface=full_camera_capture_preprocess_executor_map_render_headless"
if y26_is_board; then y26_print_command "${args[@]}"; "${args[@]}"; else y26_remote_command "${args[@]}"; fi
