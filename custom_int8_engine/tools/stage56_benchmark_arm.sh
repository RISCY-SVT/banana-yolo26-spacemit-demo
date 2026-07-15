#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    printf 'usage: %s <stage-root> <label> compatibility|low-latency <runs> <repeats> [env-file]\n' "$0" >&2
    exit 2
fi

stage_root=$1
label=$2
profile=$3
runs=$4
repeats=$5
env_file=${6:-}
binary=${Y26_STAGE56_BINARY:-$stage_root/bin/yolo26_k1x_int8}
package=${Y26_STAGE56_PACKAGE:-$stage_root/packages/stage55-control}
fixture=${Y26_STAGE56_FIXTURE:-$stage_root/fixtures/full_f0_f7/F0_nchw_f32.bin}

if [[ -n ${Y26_STAGE56_CGROUP:-} ]]; then
    printf '%s\n' "$$" | sudo -n tee "$Y26_STAGE56_CGROUP/cgroup.procs" >/dev/null
fi
if [[ -n ${Y26_STAGE56_RT_PRIORITY:-} ]]; then
    case "$Y26_STAGE56_RT_PRIORITY" in
        ''|*[!0-9]*) printf 'invalid FIFO priority: %s\n' "$Y26_STAGE56_RT_PRIORITY" >&2; exit 2 ;;
    esac
    ((Y26_STAGE56_RT_PRIORITY >= 1 && Y26_STAGE56_RT_PRIORITY <= 50)) || {
        printf 'FIFO priority must be in [1, 50]\n' >&2
        exit 2
    }
    sudo -n chrt -f -p "$Y26_STAGE56_RT_PRIORITY" "$$"
fi
if [[ -n ${Y26_STAGE56_MEMLOCK_UNLIMITED:-} ]]; then
    sudo -n prlimit --pid "$$" --memlock=unlimited:unlimited
fi

case "$label" in
    *[!A-Za-z0-9_.-]*) printf 'invalid label: %s\n' "$label" >&2; exit 2 ;;
esac
case "$profile" in
compatibility)
    unset Y26_STAGE53_SPIN_POOL Y26_STAGE55_FRAME_GATED_SPIN || true
    ;;
low-latency)
    export Y26_STAGE53_SPIN_POOL=1
    export Y26_STAGE55_FRAME_GATED_SPIN=1
    ;;
*)
    printf 'invalid profile: %s\n' "$profile" >&2
    exit 2
    ;;
esac

export Y26_STAGE54_E2C3=1
export Y26_STAGE55_E2C4=1
export Y26_STAGE55_DENSE_FAMILY_A=1
export Y26_STAGE54_DIRECT_1X1=1
export Y26_STAGE54_DENSE_PACK_RVV=1
export Y26_STAGE53_FUSED_LUT=1
export Y26_STAGE54_DEPTHWISE_V2=1
export Y26_STAGE54_DEPTHWISE_X2=1
export Y26_STAGE54_DEPTHWISE_BORDER_V2=1
export Y26_STAGE54_INPUT_RVV_V2=1
export Y26_STAGE54_INPUT_COMPACT_C3=1
export Y26_STAGE54_LUT2_RVV=1
export Y26_STAGE54_ATTENTION_V2=1
unset Y26_STAGE54_HEAD_V2 Y26_STAGE55_DEPTHWISE_E2C4 || true

if [[ -n $env_file ]]; then
    # The file is generated inside the stage root and contains export/unset statements only.
    # shellcheck disable=SC1090
    source "$env_file"
fi

ram_root=/dev/shm/y26-stage56
mkdir -p "$ram_root" "$stage_root/profiles" "$stage_root/predictions"
log="$ram_root/$label.log"
time_log="$ram_root/$label.time"
output_json="$ram_root/$label.json"
environment="$ram_root/$label.environment"
disk_before="$ram_root/$label.disk-before"
disk_after="$ram_root/$label.disk-after"
rm -f "$log" "$time_log" "$output_json" "$environment" "$disk_before" "$disk_after"

{
    printf 'utc_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'boot_id=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
    printf 'profile=%s\n' "$profile"
    printf 'runs=%s\nrepeats=%s\n' "$runs" "$repeats"
    printf 'binary_path=%s\n' "$binary"
    printf 'package_path=%s\n' "$package"
    printf 'fixture_path=%s\n' "$fixture"
    printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | awk '{print $1}')"
    printf 'package_sha256=%s\n' "$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')"
    env | LC_ALL=C sort | grep '^Y26_' || true
    for cpu in 0 1 2 3 4; do
        printf 'cpu%s_governor=' "$cpu"
        cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor" 2>/dev/null || true
        printf 'cpu%s_frequency_khz=' "$cpu"
        cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq" 2>/dev/null || true
    done
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        printf '%s=' "$zone"
        cat "$zone" 2>/dev/null || true
    done
} >"$environment"
cat /proc/diskstats >"$disk_before"

set +e
taskset -c 0-4 /usr/bin/time -v \
    "$binary" \
    --package "$package" \
    --image "$fixture" \
    --input-mode preprocessed-f32 \
    --output-json "$output_json" \
    --threads 4 --pin 0-3 --scheduler safe \
    --warmup 10 --runs "$runs" --repeats "$repeats" \
    --verify --benchmark >"$log" 2>"$time_log"
status=$?
set -e
cat /proc/diskstats >"$disk_after"
printf 'utc_end=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" \
    >>"$environment"

for file in "$log" "$time_log" "$output_json" "$environment" "$disk_before" "$disk_after"; do
    cp "$file" "$stage_root/profiles/$label.${file##*.}"
done
cp "$output_json" "$stage_root/predictions/$label.json"
rm -f "$stage_root/profiles/$label.sha256"
sha256sum "$stage_root/profiles/$label."* "$stage_root/predictions/$label.json" \
    >"$stage_root/profiles/$label.sha256"
exit "$status"
