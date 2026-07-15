#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <stage-root>\n' "$0" >&2
    exit 2
fi

stage_root=$1
inventory_root="$stage_root/profiles/system-inventory"
backup_root="$stage_root/boot/recovery-backup"
mkdir -p "$inventory_root" "$backup_root"

capture() {
    local name=$1
    shift
    {
        printf 'command:'
        printf ' %q' "$@"
        printf '\n'
        "$@"
    } >"$inventory_root/$name.stdout" 2>"$inventory_root/$name.stderr" || {
        status=$?
        printf '%s\n' "$status" >"$inventory_root/$name.exit"
        return 0
    }
    printf '0\n' >"$inventory_root/$name.exit"
}

capture_shell() {
    local name=$1
    local command=$2
    capture "$name" bash -lc "$command"
}

capture uname uname -a
capture os-release cat /etc/os-release
capture proc-cmdline cat /proc/cmdline
capture proc-interrupts cat /proc/interrupts
capture proc-softirqs cat /proc/softirqs
capture proc-schedstat cat /proc/schedstat
capture proc-vmstat cat /proc/vmstat
capture proc-meminfo cat /proc/meminfo
capture proc-cpuinfo cat /proc/cpuinfo
capture boot-id cat /proc/sys/kernel/random/boot_id
capture mount-findmnt findmnt -A
capture mount-options findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS
capture lsblk lsblk -b -o NAME,TYPE,FSTYPE,SIZE,ROTA,RO,MOUNTPOINTS
capture cgroup-mount findmnt -T /sys/fs/cgroup
capture cgroup-controllers cat /sys/fs/cgroup/cgroup.controllers
capture cgroup-subtree cat /sys/fs/cgroup/cgroup.subtree_control
capture cpu-isolated cat /sys/devices/system/cpu/isolated
capture cpu-nohz-full cat /sys/devices/system/cpu/nohz_full
capture cpu-rcu-nocbs cat /sys/devices/system/cpu/rcu_nocbs
capture workqueue-cpumask cat /sys/devices/virtual/workqueue/cpumask
capture tracefs-tracers cat /sys/kernel/tracing/available_tracers
capture tracefs-events cat /sys/kernel/tracing/available_events
capture perf-list perf list --details
capture systemd-version systemctl --version
capture service-irqbalance systemctl show irqbalance.service -p LoadState -p ActiveState -p UnitFileState
capture service-journald systemctl show systemd-journald.service -p LoadState -p ActiveState -p UnitFileState
capture service-ssh systemctl show ssh.service -p LoadState -p ActiveState -p UnitFileState
capture thermal-zones bash -lc 'for z in /sys/class/thermal/thermal_zone*; do [[ -d $z ]] || continue; printf "%s\t" "$z"; cat "$z/type" 2>/dev/null || true; cat "$z/temp" 2>/dev/null || true; done'
capture cooling-devices bash -lc 'for d in /sys/class/thermal/cooling_device*; do [[ -d $d ]] || continue; printf "%s\t" "$d"; cat "$d/type" 2>/dev/null || true; cat "$d/cur_state" 2>/dev/null || true; cat "$d/max_state" 2>/dev/null || true; done'
capture cpufreq bash -lc 'for c in /sys/devices/system/cpu/cpu[0-9]*; do [[ -d $c/cpufreq ]] || continue; printf "[%s]\n" "$c"; for f in scaling_driver scaling_governor scaling_available_governors scaling_available_frequencies scaling_cur_freq scaling_min_freq scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq affected_cpus related_cpus; do printf "%s=" "$f"; cat "$c/cpufreq/$f" 2>/dev/null || true; done; done'
capture cpuidle bash -lc 'for c in /sys/devices/system/cpu/cpu[0-9]*; do for s in "$c"/cpuidle/state*; do [[ -d $s ]] || continue; printf "[%s]\n" "$s"; for f in name desc latency residency disable usage time; do printf "%s=" "$f"; cat "$s/$f" 2>/dev/null || true; done; done; done'
capture devfreq bash -lc 'for d in /sys/class/devfreq/*; do [[ -e $d ]] || continue; printf "[%s]\n" "$d"; for f in name governor available_governors available_frequencies cur_freq min_freq max_freq target_freq; do printf "%s=" "$f"; cat "$d/$f" 2>/dev/null || true; done; done'
capture event-sources bash -lc 'for d in /sys/bus/event_source/devices/*; do [[ -d $d ]] || continue; printf "[%s]\n" "$d"; for f in type cpumask; do printf "%s=" "$f"; cat "$d/$f" 2>/dev/null || true; done; find "$d" -maxdepth 2 -type f -path "*/events/*" -print -exec cat {} \; 2>/dev/null || true; done'
capture irq-affinity bash -lc 'for d in /proc/irq/[0-9]*; do irq=${d##*/}; printf "%s\t" "$irq"; cat "$d/smp_affinity_list" 2>/dev/null || true; printf "\t"; cat "$d/effective_affinity_list" 2>/dev/null || true; done'
capture irq-actions bash -lc 'cat /proc/interrupts'
capture kernel-config bash -lc 'if [[ -r /proc/config.gz ]]; then gzip -dc /proc/config.gz; elif [[ -r /boot/config-$(uname -r) ]]; then cat /boot/config-$(uname -r); else exit 2; fi'
capture boot-tree bash -lc 'find /boot -maxdepth 4 -xdev -printf "%y\t%s\t%p\n" 2>/dev/null | sort'
capture boot-env bash -lc 'cat /boot/env_k1-x.txt'
capture boot-mount findmnt /boot
capture block-identities bash -lc 'lsblk -f; sudo -n blkid'
capture boot-dmesg bash -lc 'sudo -n dmesg | grep -Ei "U-Boot|OpenSBI|bootargs|Kernel command|Machine model|pmu|sbi" || true'
capture bootloader-env bash -lc 'command -v fw_printenv >/dev/null && fw_printenv || true; command -v grub-editenv >/dev/null && grub-editenv list || true'
capture dt-model bash -lc 'tr "\000" "\n" </sys/firmware/devicetree/base/model; tr "\000" "\n" </sys/firmware/devicetree/base/compatible'
capture live-fdt-hash bash -lc 'sha256sum /sys/firmware/fdt; stat -c "bytes=%s" /sys/firmware/fdt'
capture pmu-dt-paths bash -lc 'find /sys/firmware/devicetree/base -type f \( -path "*pmu*" -o -name "riscv,event-to-mhpmevent" -o -name "riscv,event-to-mhpmcounters" -o -name "riscv,raw-event-to-mhpmcounters" \) -print 2>/dev/null | sort'
capture pmu-dt-values bash -lc 'while IFS= read -r f; do printf "[%s]\n" "$f"; od -An -tx4 -v "$f" 2>/dev/null || od -An -tx1 -v "$f"; done < <(find /sys/firmware/devicetree/base -type f \( -name "riscv,event-to-mhpmevent" -o -name "riscv,event-to-mhpmcounters" -o -name "riscv,raw-event-to-mhpmcounters" \) -print 2>/dev/null | sort)'

while IFS= read -r path; do
    relative=${path#/}
    destination="$backup_root/$relative"
    mkdir -p "$(dirname "$destination")"
    cp -a "$path" "$destination"
done < <(find /boot -maxdepth 4 -xdev -type f 2>/dev/null | sort)

find "$backup_root" -type f -print0 | sort -z | xargs -0 -r sha256sum \
    >"$stage_root/boot/boot_profile_backup_sha256.txt"
printf 'inventory_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$inventory_root/inventory-complete.txt"
