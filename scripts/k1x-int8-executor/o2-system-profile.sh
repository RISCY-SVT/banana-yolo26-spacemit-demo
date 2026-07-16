#!/usr/bin/env bash
set -euo pipefail

readonly profile_name=y26-stage57-o2
readonly cgroup=/sys/fs/cgroup/y26-stage57-inference
readonly state_parent=${Y26_O2_STATE_PARENT:-/data/k1x-yolo26-int8-executor/state}
readonly state_dir=${Y26_O2_STATE_DIR:-$state_parent/stage57-o2}
readonly lock_file=${Y26_O2_LOCK_FILE:-$state_parent/.stage57-o2.lock}
readonly inference_cpus=0-4
readonly housekeeping_cpus=5-7
readonly workqueue_mask=e0
readonly irq_allow_pattern='nvme|end[01]|xhci|ehci|usb|mmc[0-9]|DPU_|pvrsrvkm|linlon|k1xccic|mars-cpp|feisp|thermal_zone|UART[0-9]'
readonly irq_deny_pattern='riscv-timer|riscv-pmu|IPI|pwrkey|rtc@pmic|mailbox|i2c|pinctrl'

readonly -a services=(
  cups.service
  cups-browsed.service
  bluetooth.service
  ModemManager.service
  packagekit.service
)
readonly -a units=(system.slice user.slice init.scope)
o2_apply_in_progress=no

usage() {
  cat <<'EOF'
usage: o2-system-profile.sh status
       o2-system-profile.sh dry-run
       o2-system-profile.sh apply
       o2-system-profile.sh restore
       o2-system-profile.sh restore-stale
       o2-system-profile.sh run [--timeout SECONDS] -- COMMAND [ARG...]

The O2 profile is reversible. It isolates CPU0-4 for the executor and keeps
CPU5-7 available for SSH, reviewed device IRQs, and unbound workqueues.
EOF
}

write_value() {
  local value=$1
  local path=$2
  printf '%s\n' "$value" | sudo -n tee "$path" >/dev/null
}

require_board_contract() {
  [[ $(cat /sys/devices/system/cpu/online) == 0-7 ]] || {
    echo "O2 requires exactly online CPUs 0-7" >&2
    return 1
  }
  [[ $(stat -fc %T /sys/fs/cgroup) == cgroup2fs ]] || {
    echo "O2 requires cgroup v2" >&2
    return 1
  }
  [[ -r /sys/fs/cgroup/cgroup.controllers ]] &&
    grep -qw cpuset /sys/fs/cgroup/cgroup.controllers || {
      echo "O2 requires the cgroup v2 cpuset controller" >&2
      return 1
    }
  [[ -w /data ]] || {
    echo "O2 requires writable NVMe /data" >&2
    return 1
  }
  sudo -n true || {
    echo "O2 requires non-interactive sudo for reversible system placement" >&2
    return 1
  }
}

irq_description() {
  local irq=$1
  awk -v key="$irq:" '$1 == key {$1=""; sub(/^ +/, ""); print; exit}' /proc/interrupts
}

irq_is_selected() {
  local description=$1
  [[ $description =~ $irq_allow_pattern ]] && [[ ! $description =~ $irq_deny_pattern ]]
}

write_irq_plan() {
  local output=$1
  printf 'irq\tdescription\tselected\treason\n' >"$output"
  local path irq description selected reason
  for path in /proc/irq/[0-9]*/smp_affinity_list; do
    [[ -r $path ]] || continue
    irq=${path#/proc/irq/}
    irq=${irq%%/*}
    description=$(irq_description "$irq")
    selected=no
    reason=not-reviewed
    if irq_is_selected "$description"; then
      selected=yes
      reason=reviewed-device-class
    elif [[ $description =~ $irq_deny_pattern ]]; then
      reason=critical-or-per-cpu-denylist
    fi
    printf '%s\t%s\t%s\t%s\n' "$irq" "$description" "$selected" "$reason" >>"$output"
  done
}

snapshot_state() {
  [[ ! -e $state_dir ]] || {
    echo "active or stale O2 snapshot exists: $state_dir" >&2
    echo "use status, restore, or restore-stale" >&2
    return 1
  }
  mkdir -p "$state_parent"
  local temporary
  temporary=$(mktemp -d "$state_parent/.stage57-o2-state.XXXXXX")
  trap 'sudo -n rm -rf "$temporary" 2>/dev/null || true' RETURN
  cat /sys/devices/virtual/workqueue/cpumask >"$temporary/workqueue-cpumask"
  cat /sys/fs/cgroup/cgroup.subtree_control >"$temporary/subtree-control"
  printf '%s\n' "$$" >"$temporary/owner-pid"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$temporary/applied-at"
  printf 'unit\tallowed_cpus\n' >"$temporary/allowed-cpus.tsv"
  local unit service path irq
  for unit in "${units[@]}"; do
    printf '%s\t%s\n' "$unit" \
      "$(systemctl show "$unit" -p AllowedCPUs --value 2>/dev/null || true)" \
      >>"$temporary/allowed-cpus.tsv"
  done
  printf 'service\tstate\n' >"$temporary/services.tsv"
  for service in "${services[@]}"; do
    printf '%s\t%s\n' "$service" \
      "$(systemctl is-active "$service" 2>/dev/null || true)" \
      >>"$temporary/services.tsv"
  done
  printf 'irq\taffinity\n' >"$temporary/irq-affinity.tsv"
  for path in /proc/irq/[0-9]*/smp_affinity_list; do
    [[ -r $path ]] || continue
    irq=${path#/proc/irq/}
    irq=${irq%%/*}
    printf '%s\t%s\n' "$irq" "$(cat "$path")" >>"$temporary/irq-affinity.tsv"
  done
  write_irq_plan "$temporary/irq-policy.tsv"
  chmod -R u=rwX,go=rX "$temporary"
  sudo -n chown -R root:root "$temporary"
  sudo -n mv "$temporary" "$state_dir"
  trap - RETURN
}

restore_profile() {
  [[ -d $state_dir ]] || {
    echo "O2 profile is not applied"
    return 0
  }
  local pid irq mask unit value service service_state original_subtree
  if [[ -d $cgroup ]]; then
    write_value member "$cgroup/cpuset.cpus.partition" 2>/dev/null || true
    if [[ -r $cgroup/cgroup.procs ]]; then
      while read -r pid; do
        [[ -n $pid ]] || continue
        write_value "$pid" /sys/fs/cgroup/cgroup.procs 2>/dev/null || true
      done <"$cgroup/cgroup.procs"
    fi
    sudo -n rmdir "$cgroup" 2>/dev/null || true
  fi
  if [[ -r $state_dir/irq-affinity.tsv ]]; then
    while IFS=$'\t' read -r irq mask; do
      [[ $irq != irq && -n $irq && -e /proc/irq/$irq/smp_affinity_list ]] || continue
      write_value "$mask" "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || true
    done <"$state_dir/irq-affinity.tsv"
  fi
  if [[ -r $state_dir/workqueue-cpumask ]]; then
    write_value "$(cat "$state_dir/workqueue-cpumask")" \
      /sys/devices/virtual/workqueue/cpumask 2>/dev/null || true
  fi
  if [[ -r $state_dir/allowed-cpus.tsv ]]; then
    while IFS=$'\t' read -r unit value; do
      [[ $unit != unit ]] || continue
      sudo -n systemctl set-property --runtime "$unit" "AllowedCPUs=$value" \
        >/dev/null 2>&1 || true
    done <"$state_dir/allowed-cpus.tsv"
  fi
  if [[ -r $state_dir/services.tsv ]]; then
    while IFS=$'\t' read -r service service_state; do
      [[ $service != service ]] || continue
      if [[ $service_state == active ]]; then
        sudo -n systemctl start "$service" >/dev/null 2>&1 || true
      fi
    done <"$state_dir/services.tsv"
  fi
  if [[ -r $state_dir/subtree-control ]]; then
    original_subtree=$(cat "$state_dir/subtree-control")
    if [[ $original_subtree != *cpuset* ]]; then
      write_value -cpuset /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
    fi
  fi
  sudo -n rm -rf "$state_dir"
  echo "O2 profile restored"
}

apply_profile() {
  require_board_contract
  snapshot_state
  o2_apply_in_progress=yes
  trap 'status=$?; if [[ ${o2_apply_in_progress:-no} == yes ]]; then restore_profile || true; fi; exit $status' EXIT
  write_value +cpuset /sys/fs/cgroup/cgroup.subtree_control
  sudo -n mkdir "$cgroup"
  write_value "$(cat /sys/fs/cgroup/cpuset.mems.effective)" "$cgroup/cpuset.mems"
  write_value "$inference_cpus" "$cgroup/cpuset.cpus"
  if [[ -e $cgroup/cpuset.cpus.exclusive ]]; then
    write_value "$inference_cpus" "$cgroup/cpuset.cpus.exclusive"
  fi
  write_value isolated "$cgroup/cpuset.cpus.partition"
  [[ $(cat "$cgroup/cpuset.cpus.effective") == "$inference_cpus" ]] || {
    echo "effective inference cpuset is not $inference_cpus" >&2
    return 1
  }
  [[ $(cat "$cgroup/cpuset.cpus.partition") == isolated ]] || {
    echo "cpuset partition did not become isolated" >&2
    return 1
  }
  local unit service irq description target slot=0
  for unit in "${units[@]}"; do
    sudo -n systemctl set-property --runtime "$unit" "AllowedCPUs=$housekeeping_cpus"
  done
  while IFS=$'\t' read -r irq description selected _reason; do
    [[ $irq != irq && $selected == yes ]] || continue
    [[ -e /proc/irq/$irq/smp_affinity_list ]] || continue
    target=$((5 + slot % 3))
    slot=$((slot + 1))
    if ! write_value "$target" "/proc/irq/$irq/smp_affinity_list" 2>/dev/null; then
      printf 'warning: IRQ %s is reviewed but not movable: %s\n' "$irq" "$description" >&2
    fi
  done <"$state_dir/irq-policy.tsv"
  write_value "$workqueue_mask" /sys/devices/virtual/workqueue/cpumask
  for service in "${services[@]}"; do
    [[ $(systemctl is-active "$service" 2>/dev/null || true) == active ]] || continue
    if ! sudo -n systemctl stop "$service"; then
      printf 'warning: optional service could not be stopped: %s\n' "$service" >&2
    fi
  done
  o2_apply_in_progress=no
  trap - EXIT
  echo "O2 profile applied"
}

status_profile() {
  printf 'profile\t%s\n' "$profile_name"
  printf 'snapshot\t%s\n' "$([[ -d $state_dir ]] && echo present || echo absent)"
  printf 'cgroup\t%s\n' "$([[ -d $cgroup ]] && echo present || echo absent)"
  if [[ -d $cgroup ]]; then
    printf 'cpus_effective\t%s\n' "$(cat "$cgroup/cpuset.cpus.effective")"
    printf 'partition\t%s\n' "$(cat "$cgroup/cpuset.cpus.partition")"
    printf 'processes\t%s\n' "$(tr '\n' ',' <"$cgroup/cgroup.procs" | sed 's/,$//')"
  fi
  printf 'workqueue_cpumask\t%s\n' "$(cat /sys/devices/virtual/workqueue/cpumask)"
}

dry_run() {
  require_board_contract
  local plan
  plan=$(mktemp)
  trap 'rm -f "$plan"' RETURN
  write_irq_plan "$plan"
  echo "inference_cpus=$inference_cpus"
  echo "housekeeping_cpus=$housekeeping_cpus"
  echo "workqueue_cpumask=$workqueue_mask"
  awk -F '\t' '$3 == "yes" {print "irq=" $1 "\t" $2}' "$plan"
  trap - RETURN
  rm -f "$plan"
}

run_profile() {
  local timeout_seconds=0
  if [[ ${1:-} == --timeout ]]; then
    [[ $# -ge 3 ]] || { usage >&2; return 2; }
    timeout_seconds=$2
    shift 2
  fi
  [[ ${1:-} == -- ]] || { usage >&2; return 2; }
  shift
  [[ $# -gt 0 ]] || { usage >&2; return 2; }
  [[ $timeout_seconds =~ ^[0-9]+$ ]] || {
    echo "timeout must be a non-negative integer" >&2
    return 2
  }
  apply_profile
  local child=0 command_status=0
  cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    if (( child > 0 )); then
      kill -TERM "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
    fi
    restore_profile || true
    exit "$status"
  }
  interrupted() {
    local signal=$1
    command_status=$((128 + signal))
    exit "$command_status"
  }
  trap cleanup EXIT
  trap 'interrupted 2' INT
  trap 'interrupted 15' TERM
  trap 'interrupted 1' HUP
  (
    exec 9>&-
    write_value "$BASHPID" "$cgroup/cgroup.procs"
    if (( timeout_seconds > 0 )); then
      exec timeout --preserve-status --signal=TERM --kill-after=5 "$timeout_seconds" "$@"
    fi
    exec "$@"
  ) &
  child=$!
  set +e
  wait "$child"
  command_status=$?
  set -e
  child=0
  restore_profile
  trap - EXIT INT TERM HUP
  return "$command_status"
}

main() {
  [[ $# -ge 1 ]] || { usage >&2; return 2; }
  mkdir -p "$state_parent"
  exec 9>"$lock_file"
  flock -n 9 || {
    echo "another O2 lifecycle operation holds $lock_file" >&2
    return 1
  }
  local action=$1
  shift
  case "$action" in
    status) [[ $# == 0 ]] || { usage >&2; return 2; }; status_profile ;;
    dry-run) [[ $# == 0 ]] || { usage >&2; return 2; }; dry_run ;;
    apply) [[ $# == 0 ]] || { usage >&2; return 2; }; apply_profile ;;
    restore) [[ $# == 0 ]] || { usage >&2; return 2; }; restore_profile ;;
    restore-stale) [[ $# == 0 ]] || { usage >&2; return 2; }; restore_profile ;;
    run) run_profile "$@" ;;
    --help|-h|help) usage ;;
    *) usage >&2; return 2 ;;
  esac
}

main "$@"
