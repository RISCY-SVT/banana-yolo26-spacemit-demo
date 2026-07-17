#!/usr/bin/env bash
set -euo pipefail

action=${Y26_CAMERA_IRQ_ACTION:-xhci-hcd:usb2}
camera_cpu=${Y26_CAMERA_CPU:-5}
state_root=${Y26_CAMERA_PROFILE_STATE_ROOT:-/data/y26-camera-profile}
state_file=$state_root/irq-state.tsv
lock_file=$state_root/profile.lock
child_pid=

usage() {
  cat <<'EOF'
usage: camera-system-profile.sh status
       camera-system-profile.sh restore-stale
       camera-system-profile.sh run -- COMMAND [ARG ...]

Temporarily pins the reviewed camera xHCI IRQ to CPU5 while COMMAND runs.
The original affinity is restored on normal exit, failure, INT, TERM, or HUP.
EOF
}

find_irq() {
  mapfile -t matches < <(
    awk -v action="$action" 'index($0, action) { gsub(":", "", $1); print $1 }' \
      /proc/interrupts
  )
  if ((${#matches[@]} != 1)) || [[ ! ${matches[0]} =~ ^[0-9]+$ ]]; then
    echo "expected one numeric IRQ for action '$action', found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

validate_cpu() {
  local cpu_path
  [[ $camera_cpu =~ ^[0-9]+$ ]] || {
    echo "invalid camera CPU: $camera_cpu" >&2
    return 1
  }
  cpu_path=/sys/devices/system/cpu/cpu$camera_cpu
  [[ -d $cpu_path && (! -f $cpu_path/online || $(<"$cpu_path/online") == 1) ]] || {
    echo "camera CPU $camera_cpu is not online" >&2
    return 1
  }
}

restore() {
  [[ -f $state_file ]] || return 0
  local irq original
  IFS=$'\t' read -r irq original <"$state_file"
  if [[ $irq =~ ^[0-9]+$ && -n $original && -e /proc/irq/$irq/smp_affinity_list ]]; then
    printf '%s\n' "$original" | sudo tee "/proc/irq/$irq/smp_affinity_list" >/dev/null
    local restored
    restored=$(<"/proc/irq/$irq/smp_affinity_list")
    if [[ $restored != "$original" ]]; then
      echo "camera IRQ restore mismatch: expected $original, got $restored" >&2
      return 1
    fi
  fi
  rm -f "$state_file"
}

on_signal() {
  local signal=$1 code=1
  case $signal in
    INT) code=130 ;;
    TERM) code=143 ;;
    HUP) code=129 ;;
  esac
  if [[ -n $child_pid ]]; then
    kill -s "$signal" "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=
  fi
  restore
  trap - EXIT
  exit "$code"
}

case ${1:-} in
  status)
    validate_cpu
    irq=$(find_irq)
    printf 'action=%s irq=%s requested_affinity=%s effective_affinity=%s camera_cpu=%s state=%s\n' \
      "$action" "$irq" "$(<"/proc/irq/$irq/smp_affinity_list")" \
      "$(<"/proc/irq/$irq/effective_affinity_list")" "$camera_cpu" \
      "$([[ -f $state_file ]] && echo active || echo inactive)"
    ;;
  restore-stale)
    mkdir -p "$state_root"
    exec 9>"$lock_file"
    flock -n 9 || { echo "camera system profile is active" >&2; exit 1; }
    if [[ -f $state_file ]]; then
      restore
      echo "stale camera system profile restored"
    else
      echo "no stale camera system profile state"
    fi
    ;;
  run)
    shift
    [[ ${1:-} == -- ]] || { usage >&2; exit 2; }
    shift
    (($#)) || { usage >&2; exit 2; }
    validate_cpu
    mkdir -p "$state_root"
    exec 9>"$lock_file"
    flock -n 9 || { echo "camera system profile is already active" >&2; exit 1; }
    [[ ! -f $state_file ]] || {
      echo "stale camera profile state exists: $state_file" >&2
      exit 1
    }
    irq=$(find_irq)
    original=$(<"/proc/irq/$irq/smp_affinity_list")
    state_tmp=$state_file.$$
    printf '%s\t%s\n' "$irq" "$original" >"$state_tmp"
    chmod 0600 "$state_tmp"
    mv "$state_tmp" "$state_file"
    trap restore EXIT
    trap 'on_signal INT' INT
    trap 'on_signal TERM' TERM
    trap 'on_signal HUP' HUP
    printf '%s\n' "$camera_cpu" | sudo tee "/proc/irq/$irq/smp_affinity_list" >/dev/null
    effective=$(<"/proc/irq/$irq/effective_affinity_list")
    [[ $effective == "$camera_cpu" ]] || {
      echo "camera IRQ $irq effective affinity is $effective, expected $camera_cpu" >&2
      exit 1
    }
    "$@" &
    child_pid=$!
    set +e
    wait "$child_pid"
    status=$?
    set -e
    child_pid=
    restore
    trap - EXIT INT TERM HUP
    exit "$status"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
