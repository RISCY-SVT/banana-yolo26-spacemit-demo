#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 IMAGE [ANNOTATED_OUTPUT] [extra y26_k1x_demo arguments]"
  exit 0
fi
(( $# >= 1 )) || { echo "image path is required" >&2; exit 2; }
image=$1
shift
output=${1:-/data/Screenshots/yolo26-stage58-image.png}
if (($#)); then shift; fi
root=$Y26_BOARD_RELEASE_ROOT
args=("$root/bin/y26_k1x_demo" --package "$root/package" --labels "$root/labels/coco80.txt"
      --expected-manifest-sha256 "$Y26_EXPECTED_MANIFEST_SHA256" --source "image:$image"
      --profile low-latency --headless --save-frame "$output" "$@")
if y26_is_board; then
  y26_require_release "$root"
  y26_print_command "${args[@]}"
  "${args[@]}"
else
  y26_print_command ssh "$Y26_BOARD_TARGET" "${args[@]}"
  y26_remote_command "${args[@]}"
fi
