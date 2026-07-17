#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/y26_executor_common.sh"

[[ ${1:-} != --help && ${1:-} != -h ]] || {
  echo "usage: $0 [PID]  # inspect Stage59 demo threads, cgroup, and IRQ placement"
  exit 0
}
inspect='set -euo pipefail
pid=${1:-$(pgrep -n y26_k1x_demo || true)}
[[ -n $pid ]] || { echo "y26_k1x_demo is not running" >&2; exit 1; }
echo "pid=$pid"
ps -L -p "$pid" -o pid,tid,psr,cls,rtprio,stat,comm
grep -E "Cpus_allowed_list|voluntary_ctxt_switches|nonvoluntary_ctxt_switches" "/proc/$pid/status"
cat "/proc/$pid/cgroup"
for task in /proc/$pid/task/*; do
  printf "tid=%s affinity=" "${task##*/}"
  taskset -pc "${task##*/}" 2>/dev/null | sed "s/.*: //"
done
grep -Ei "nvme|xhci|usb|video|ccic|riscv-timer|IPI" /proc/interrupts || true
if [[ -d /sys/fs/cgroup/y26-inference ]]; then
  find /sys/fs/cgroup/y26-inference -maxdepth 1 -type f -name "cpuset.*" -print -exec cat {} \;
fi'
if y26_is_board; then bash -c "$inspect" _ "${1:-}"; else y26_remote_command bash -c "$inspect" _ "${1:-}"; fi
