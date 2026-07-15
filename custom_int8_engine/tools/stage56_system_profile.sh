#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s apply|restore <stage-root> O0|O1|O2|O3\n' "$0" >&2
    exit 2
fi

action=$1
stage_root=$2
arm=$3
state_root="$stage_root/profiles/system-state-before"
profile_root="$stage_root/cgroup/runtime-profile"
inference_cgroup=/sys/fs/cgroup/y26-stage56-inference

write_value() {
    local value=$1
    local path=$2
    printf '%s\n' "$value" | sudo -n tee "$path" >/dev/null
}

restore_profile() {
    if [[ -d $inference_cgroup ]]; then
        write_value member "$inference_cgroup/cpuset.cpus.partition" || true
        sudo -n rmdir "$inference_cgroup" 2>/dev/null || true
    fi
    if [[ -r $profile_root/allowed-cpus-before.tsv ]]; then
        while IFS=$'\t' read -r unit value; do
            [[ -n $unit ]] || continue
            if [[ -n $value ]]; then
                sudo -n systemctl set-property --runtime "$unit" "AllowedCPUs=$value" \
                    2>/dev/null || true
            else
                sudo -n systemctl set-property --runtime "$unit" AllowedCPUs= \
                    2>/dev/null || true
            fi
        done <"$profile_root/allowed-cpus-before.tsv"
    fi
    if [[ -r $profile_root/services-before.tsv ]]; then
        while IFS=$'\t' read -r service state; do
            [[ $state == active ]] && sudo -n systemctl start "$service" >/dev/null || true
        done <"$profile_root/services-before.tsv"
    fi
    "$stage_root/bin/stage56_runtime_state.sh" restore "$state_root"
    if [[ -r $profile_root/subtree-control-before ]]; then
        before=$(cat "$profile_root/subtree-control-before")
        if [[ $before != *cpuset* ]]; then
            write_value -cpuset /sys/fs/cgroup/cgroup.subtree_control || true
        fi
    fi
}

create_isolated_partition() {
    write_value +cpuset /sys/fs/cgroup/cgroup.subtree_control
    sudo -n mkdir -p "$inference_cgroup"
    write_value "$(cat /sys/fs/cgroup/cpuset.mems.effective)" \
        "$inference_cgroup/cpuset.mems"
    write_value 0-4 "$inference_cgroup/cpuset.cpus"
    if [[ -e $inference_cgroup/cpuset.cpus.exclusive ]]; then
        write_value 0-4 "$inference_cgroup/cpuset.cpus.exclusive"
    fi
    write_value isolated "$inference_cgroup/cpuset.cpus.partition"
    for unit in system.slice user.slice init.scope; do
        sudo -n systemctl set-property --runtime "$unit" AllowedCPUs=5-7
    done
    {
        printf 'partition='; cat "$inference_cgroup/cpuset.cpus.partition"
        printf 'cpus='; cat "$inference_cgroup/cpuset.cpus"
        printf 'effective='; cat "$inference_cgroup/cpuset.cpus.effective"
        printf 'exclusive_effective='; cat "$inference_cgroup/cpuset.cpus.exclusive.effective" 2>/dev/null || true
        printf 'mems='; cat "$inference_cgroup/cpuset.mems.effective"
    } >"$profile_root/$arm-cgroup-state.txt"
}

snapshot_systemd_affinity() {
    : >"$profile_root/allowed-cpus-before.tsv"
    for unit in system.slice user.slice init.scope; do
        value=$(systemctl show "$unit" -p AllowedCPUs --value 2>/dev/null || true)
        printf '%s\t%s\n' "$unit" "$value" >>"$profile_root/allowed-cpus-before.tsv"
    done
}

move_housekeeping_work() {
    : >"$profile_root/$arm-irq-after.tsv"
    mapfile -t interrupts < <(awk -F: '/^[[:space:]]*[0-9]+:/ {gsub(/[[:space:]]/, "", $1); print $1}' /proc/interrupts)
    local slot=0
    for irq in "${interrupts[@]}"; do
        [[ $irq == 11 || $irq == 106 ]] && continue
        path="/proc/irq/$irq/smp_affinity_list"
        [[ -e $path ]] || continue
        cpu=$((5 + slot % 3))
        slot=$((slot + 1))
        if write_value "$cpu" "$path" 2>/dev/null; then
            result=written
        else
            result=unsupported
        fi
        printf '%s\t%s\t%s\t%s\n' "$irq" "$cpu" "$result" \
            "$(cat "/proc/irq/$irq/effective_affinity_list" 2>/dev/null || true)" \
            >>"$profile_root/$arm-irq-after.tsv"
    done
    write_value e0 /sys/devices/virtual/workqueue/cpumask

    : >"$profile_root/services-before.tsv"
    for service in cups.service cups-browsed.service bluetooth.service \
                   ModemManager.service packagekit.service; do
        state=$(systemctl is-active "$service" 2>/dev/null || true)
        printf '%s\t%s\n' "$service" "$state" >>"$profile_root/services-before.tsv"
        [[ $state == active ]] && sudo -n systemctl stop "$service" >/dev/null || true
    done
}

lock_supported_cpu_frequency() {
    for cpu in /sys/devices/system/cpu/cpu[0-7]; do
        [[ -d $cpu/cpufreq ]] || continue
        maximum=$(cat "$cpu/cpufreq/cpuinfo_max_freq")
        write_value performance "$cpu/cpufreq/scaling_governor"
        write_value "$maximum" "$cpu/cpufreq/scaling_max_freq"
        write_value "$maximum" "$cpu/cpufreq/scaling_min_freq"
    done
}

case "$action" in
restore)
    restore_profile
    ;;
apply)
    [[ -r $state_root/files.tsv ]] || {
        printf 'missing baseline state: %s\n' "$state_root" >&2
        exit 1
    }
    mkdir -p "$profile_root"
    cat /sys/fs/cgroup/cgroup.subtree_control >"$profile_root/subtree-control-before"
    snapshot_systemd_affinity
    restore_profile
    case "$arm" in
        O0) ;;
        O1) create_isolated_partition ;;
        O2) create_isolated_partition; move_housekeeping_work ;;
        O3) create_isolated_partition; move_housekeeping_work; lock_supported_cpu_frequency ;;
        *) printf 'unsupported runtime arm: %s\n' "$arm" >&2; exit 2 ;;
    esac
    ;;
*)
    printf 'unsupported action: %s\n' "$action" >&2
    exit 2
    ;;
esac
