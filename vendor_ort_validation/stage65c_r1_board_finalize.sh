#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_R1_BOARD_ROOT:?STAGE65C_R1_BOARD_ROOT is required}"
root=$STAGE65C_R1_BOARD_ROOT
stage_id=${root##*/}

snapshot() {
  {
    printf 'field\tvalue\n'
    printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname\t%s\n' "$(hostname)"
    printf 'boot_id\t%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
    printf 'kernel\t%s\n' "$(uname -a)"
    printf 'device_model\t%s\n' "$(tr -d '\000' </proc/device-tree/model 2>/dev/null || true)"
    printf 'device_serial\t%s\n' "$(tr -d '\000' </proc/device-tree/serial-number 2>/dev/null || true)"
    printf 'allowed_cpu_list\t%s\n' "$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    printf 'memory\t%s\n' "$(awk '/MemTotal|MemAvailable/ {printf "%s=%s%s ", $1, $2, $3}' /proc/meminfo)"
    printf 'data_mount\t%s\n' "$(findmnt -T /data -no TARGET,SOURCE,FSTYPE,OPTIONS)"
    printf 'root_mount\t%s\n' "$(findmnt -T / -no TARGET,SOURCE,FSTYPE,OPTIONS)"
    printf 'data_free_bytes\t%s\n' "$(df -PB1 /data | awk 'NR==2 {print $4}')"
    find /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor -type f -print0 2>/dev/null \
      | sort -z \
      | xargs -0 -r -n1 sh -c 'printf "governor\t%s=%s\n" "$1" "$(cat "$1")"' sh
    find /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq -type f -print0 2>/dev/null \
      | sort -z \
      | xargs -0 -r -n1 sh -c 'printf "frequency_khz\t%s=%s\n" "$1" "$(cat "$1")"' sh
    find /sys/class/thermal/thermal_zone*/temp -type f -print0 2>/dev/null \
      | sort -z \
      | xargs -0 -r -n1 sh -c 'printf "temperature_millic\t%s=%s\n" "$1" "$(cat "$1")"' sh
  }
}

snapshot >"$root/state/system_state_after.tsv"
{
  printf 'surface\tvalue\n'
  printf 'stage_root_mount\t%s\n' "$(findmnt -T "$root" -no SOURCE,FSTYPE,OPTIONS)"
  printf 'root_mount\t%s\n' "$(findmnt -T / -no SOURCE,FSTYPE,OPTIONS)"
  printf 'emmc_stage_path_count\t%s\n' "$(find / -xdev -path "*$stage_id*" -print 2>/dev/null | wc -l)"
  printf 'active_stage_process_count\t%s\n' "$({ ps -eo args= | grep -F "$stage_id" | grep -v -E 'grep|stage65c_r1_board_finalize' || true; } | wc -l)"
} >"$root/state/storage_write_audit_after.tsv"

emmc_count=$(awk -F '\t' '$1 == "emmc_stage_path_count" {print $2}' "$root/state/storage_write_audit_after.tsv")
process_count=$(awk -F '\t' '$1 == "active_stage_process_count" {print $2}' "$root/state/storage_write_audit_after.tsv")
[[ $emmc_count == 0 ]] || { printf 'eMMC Stage paths detected: %s\n' "$emmc_count" >&2; exit 2; }
[[ $process_count == 0 ]] || { printf 'active Stage processes detected: %s\n' "$process_count" >&2; exit 2; }

printf 'stage65c_r1_board_finalize status=pass emmc_paths=0 active_processes=0\n'
