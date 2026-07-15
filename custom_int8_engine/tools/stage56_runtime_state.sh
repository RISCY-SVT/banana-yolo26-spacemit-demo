#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s snapshot|restore <state-dir>\n' "$0" >&2
    exit 2
fi

action=$1
state_dir=$2
mkdir -p "$state_dir"

snapshot_file() {
    local path=$1
    local name=${path#/}
    name=${name//\//__}
    if [[ -r $path ]]; then
        cat "$path" >"$state_dir/$name"
        printf '%s\t%s\n' "$path" "$name" >>"$state_dir/files.tsv"
    fi
}

case "$action" in
snapshot)
    : >"$state_dir/files.tsv"
    snapshot_file /sys/devices/virtual/workqueue/cpumask
    snapshot_file /proc/sys/kernel/perf_event_paranoid
    snapshot_file /proc/sys/kernel/kptr_restrict
    for path in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor \
                /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq \
                /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq \
                /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state*/disable \
                /sys/class/devfreq/*/governor \
                /sys/class/devfreq/*/min_freq \
                /sys/class/devfreq/*/max_freq \
                /proc/irq/[0-9]*/smp_affinity_list; do
        [[ -e $path ]] && snapshot_file "$path"
    done
    for service in irqbalance.service systemd-journald.service ssh.service; do
        printf '%s\t%s\n' "$service" "$(systemctl is-active "$service" 2>/dev/null || true)" \
            >>"$state_dir/services.tsv"
    done
    printf '%s\n' "$(cat /proc/sys/kernel/random/boot_id)" >"$state_dir/boot_id"
    ;;
restore)
    while IFS=$'\t' read -r path name; do
        [[ -n $path && -r $state_dir/$name && -e $path ]] || continue
        value=$(cat "$state_dir/$name")
        printf '%s' "$value" | sudo -n tee "$path" >/dev/null || true
    done <"$state_dir/files.tsv"
    if [[ -r $state_dir/services.tsv ]]; then
        while IFS=$'\t' read -r service state; do
            case "$state" in
                active) sudo -n systemctl start "$service" || true ;;
                inactive|failed) sudo -n systemctl stop "$service" || true ;;
            esac
        done <"$state_dir/services.tsv"
    fi
    ;;
*)
    printf 'unsupported action: %s\n' "$action" >&2
    exit 2
    ;;
esac
