#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 apply|restore [STATE_DIR]" >&2
  exit 2
fi

action=$1
state_dir=${2:-/data/k1x-yolo26-int8-executor/state/stage56-o2}
cgroup=/sys/fs/cgroup/y26-stage56-inference
services=(cups.service cups-browsed.service bluetooth.service ModemManager.service packagekit.service)
units=(system.slice user.slice init.scope)

write_value() {
  local value=$1
  local path=$2
  printf '%s\n' "$value" | sudo -n tee "$path" >/dev/null
}

snapshot() {
  mkdir -p "$state_dir"
  [[ ! -e $state_dir/applied ]] || {
    echo "refusing to replace an active Stage56 system-state snapshot" >&2
    exit 1
  }
  cat /sys/devices/virtual/workqueue/cpumask >"$state_dir/workqueue-cpumask"
  cat /sys/fs/cgroup/cgroup.subtree_control >"$state_dir/subtree-control"
  : >"$state_dir/allowed-cpus.tsv"
  for unit in "${units[@]}"; do
    printf '%s\t%s\n' "$unit" \
      "$(systemctl show "$unit" -p AllowedCPUs --value 2>/dev/null || true)" \
      >>"$state_dir/allowed-cpus.tsv"
  done
  : >"$state_dir/services.tsv"
  for service in "${services[@]}"; do
    printf '%s\t%s\n' "$service" \
      "$(systemctl is-active "$service" 2>/dev/null || true)" \
      >>"$state_dir/services.tsv"
  done
  : >"$state_dir/irq-affinity.tsv"
  for path in /proc/irq/[0-9]*/smp_affinity_list; do
    [[ -r $path ]] || continue
    irq=${path#/proc/irq/}
    irq=${irq%%/*}
    printf '%s\t%s\n' "$irq" "$(cat "$path")" >>"$state_dir/irq-affinity.tsv"
  done
  date -u +%Y-%m-%dT%H:%M:%SZ >"$state_dir/applied"
}

restore() {
  [[ -d $state_dir ]] || return 0
  if [[ -d $cgroup ]]; then
    write_value member "$cgroup/cpuset.cpus.partition" || true
    if [[ -r $cgroup/cgroup.procs ]]; then
      while read -r pid; do
        [[ -n $pid ]] || continue
        write_value "$pid" /sys/fs/cgroup/cgroup.procs || true
      done <"$cgroup/cgroup.procs"
    fi
    sudo -n rmdir "$cgroup" 2>/dev/null || true
  fi
  if [[ -r $state_dir/irq-affinity.tsv ]]; then
    while IFS=$'\t' read -r irq mask; do
      [[ -n $irq && -e /proc/irq/$irq/smp_affinity_list ]] || continue
      write_value "$mask" "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || true
    done <"$state_dir/irq-affinity.tsv"
  fi
  if [[ -r $state_dir/workqueue-cpumask ]]; then
    write_value "$(cat "$state_dir/workqueue-cpumask")" \
      /sys/devices/virtual/workqueue/cpumask || true
  fi
  if [[ -r $state_dir/allowed-cpus.tsv ]]; then
    while IFS=$'\t' read -r unit value; do
      if [[ -n $value ]]; then
        sudo -n systemctl set-property --runtime "$unit" "AllowedCPUs=$value" \
          >/dev/null 2>&1 || true
      else
        sudo -n systemctl set-property --runtime "$unit" AllowedCPUs= \
          >/dev/null 2>&1 || true
      fi
    done <"$state_dir/allowed-cpus.tsv"
  fi
  if [[ -r $state_dir/services.tsv ]]; then
    while IFS=$'\t' read -r service state; do
      [[ $state == active ]] && sudo -n systemctl start "$service" >/dev/null 2>&1 || true
    done <"$state_dir/services.tsv"
  fi
  if [[ -r $state_dir/subtree-control ]] && \
     [[ $(cat "$state_dir/subtree-control") != *cpuset* ]]; then
    write_value -cpuset /sys/fs/cgroup/cgroup.subtree_control || true
  fi
  rm -f "$state_dir/applied"
}

apply() {
  snapshot
  trap 'status=$?; if (( status != 0 )); then restore; fi; exit "$status"' EXIT
  write_value +cpuset /sys/fs/cgroup/cgroup.subtree_control
  sudo -n mkdir -p "$cgroup"
  write_value "$(cat /sys/fs/cgroup/cpuset.mems.effective)" "$cgroup/cpuset.mems"
  write_value 0-4 "$cgroup/cpuset.cpus"
  [[ ! -e $cgroup/cpuset.cpus.exclusive ]] || write_value 0-4 "$cgroup/cpuset.cpus.exclusive"
  write_value isolated "$cgroup/cpuset.cpus.partition"
  for unit in "${units[@]}"; do
    sudo -n systemctl set-property --runtime "$unit" AllowedCPUs=5-7
  done
  slot=0
  while IFS=$'\t' read -r irq _; do
    [[ $irq == 11 || $irq == 106 ]] && continue
    path=/proc/irq/$irq/smp_affinity_list
    [[ -e $path ]] || continue
    cpu=$((5 + slot % 3))
    slot=$((slot + 1))
    write_value "$cpu" "$path" 2>/dev/null || true
  done <"$state_dir/irq-affinity.tsv"
  write_value e0 /sys/devices/virtual/workqueue/cpumask
  for service in "${services[@]}"; do
    [[ $(systemctl is-active "$service" 2>/dev/null || true) == active ]] || continue
    sudo -n systemctl stop "$service"
  done
  trap - EXIT
}

case "$action" in
  apply) apply ;;
  restore) restore ;;
  *) echo "unknown action: $action" >&2; exit 2 ;;
esac
