#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

camera=${1:-/dev/v4l/by-id/usb-JQ-FAY-220422_BBA_USB_CAMERA_01.00.00-video-index0}
if [[ $camera == --help || $camera == -h ]]; then
  echo "usage: $0 [CAMERA_NODE]"
  exit 0
fi
if y26_is_board; then
  v4l2-ctl --list-devices
  v4l2-ctl -d "$camera" --all
  v4l2-ctl -d "$camera" --list-formats-ext
else
  y26_remote_command bash -c 'v4l2-ctl --list-devices; v4l2-ctl -d "$1" --all; v4l2-ctl -d "$1" --list-formats-ext' _ "$camera"
fi
