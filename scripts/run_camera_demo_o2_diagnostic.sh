#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

[[ ${1:-} != --help && ${1:-} != -h ]] || {
  echo "usage: $0 [extra y26_k1x_demo arguments]"
  echo "Diagnostic only: applies the reversible executor O2 profile around the camera demo."
  exit 0
}
root=$Y26_BOARD_RELEASE_ROOT
run_board() {
  y26_require_release "$root"
  y26_prepare_gui_env
  local args=("$root/bin/y26_k1x_demo" --package "$root/package" --labels "$root/labels/coco80.txt"
    --expected-manifest-sha256 "$Y26_EXPECTED_MANIFEST_SHA256" --source camera:auto
    --camera-width "${Y26_CAMERA_WIDTH:-1280}" --camera-height "${Y26_CAMERA_HEIGHT:-720}"
    --camera-fps "${Y26_CAMERA_FPS:-60}" --camera-fourcc "${Y26_CAMERA_FOURCC:-MJPG}"
    --profile low-latency-dedicated --flow latest-frame --record-mode async "$@")
  y26_print_command "$root/scripts/o2-system-profile.sh" run -- "${args[@]}"
  "$root/scripts/o2-system-profile.sh" run -- "${args[@]}"
}
if y26_is_board; then
  run_board "$@"
else
  functions=$(declare -f y26_require_release y26_prepare_gui_env y26_print_command run_board)
  quoted_args=
  if (($#)); then printf -v quoted_args '%q ' "$@"; fi
  ssh -t "$Y26_BOARD_TARGET" \
    "Y26_BOARD_RELEASE_ROOT=$(printf %q "$root") Y26_EXPECTED_MANIFEST_SHA256=$Y26_EXPECTED_MANIFEST_SHA256 bash -s -- $quoted_args" \
    <<<"set -euo pipefail
$functions
root=\$Y26_BOARD_RELEASE_ROOT
run_board \"\$@\""
fi
