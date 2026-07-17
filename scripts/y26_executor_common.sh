#!/usr/bin/env bash
set -euo pipefail

Y26_REPO_ROOT=${Y26_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
release_env=$Y26_REPO_ROOT/config/release.env
[[ -f $release_env ]] || { echo "missing release identity: $release_env" >&2; return 1 2>/dev/null || exit 1; }
# shellcheck source=../config/release.env
source "$release_env"
if [[ -x $Y26_REPO_ROOT/bin/y26_k1x_demo && -d $Y26_REPO_ROOT/package ]]; then
  y26_default_release_root=$Y26_REPO_ROOT
else
  y26_default_release_root=$Y26_RUNTIME_RELEASE_ROOT
fi
Y26_RELEASE_ROOT=${Y26_RELEASE_ROOT:-$y26_default_release_root}
Y26_BOARD_TARGET=${Y26_BOARD_TARGET:-svt@banana}
if [[ $(uname -m) == riscv64 && -d /data && -x $Y26_RELEASE_ROOT/bin/y26_k1x_demo ]]; then
  y26_default_board_root=$Y26_RELEASE_ROOT
else
  y26_default_board_root=$Y26_BOARD_INSTALL_ROOT
fi
Y26_BOARD_RELEASE_ROOT=${Y26_BOARD_RELEASE_ROOT:-$y26_default_board_root}
y26_is_board() {
  [[ $(uname -m) == riscv64 && -d /data ]]
}

y26_require_release() {
  local root=$1
  [[ -x $root/bin/y26_k1x_demo ]] || { echo "missing demo: $root/bin/y26_k1x_demo" >&2; return 1; }
  [[ -d $root/package ]] || { echo "missing prepared package: $root/package" >&2; return 1; }
  [[ -f $root/labels/coco80.txt ]] || { echo "missing labels: $root/labels/coco80.txt" >&2; return 1; }
}

y26_print_command() {
  printf 'effective command:'
  printf ' %q' "$@"
  printf '\n'
}

y26_remote_command() {
  local command
  printf -v command '%q ' "$@"
  ssh "$Y26_BOARD_TARGET" "$command"
}

y26_prepare_gui_env() {
  export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
  export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
  if [[ -S $XDG_RUNTIME_DIR/wayland-0 ]]; then
    export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
    export XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-wayland}
  fi
  if pgrep -x Xwayland >/dev/null 2>&1; then
    export DISPLAY=${DISPLAY:-:0}
    if [[ -z ${XAUTHORITY:-} ]]; then
      XAUTHORITY=$(pgrep -a Xwayland | sed -n 's/.* -auth \([^ ]*\).*/\1/p' | head -1)
      export XAUTHORITY
    fi
  fi
}
