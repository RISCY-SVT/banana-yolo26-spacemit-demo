#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    printf 'usage: %s <stage-root> <binary> <package> <fixture> <runs>\n' "$0" >&2
    exit 2
fi

stage_root=$1
binary=$2
package=$3
fixture=$4
runs=$5
trace_root=/sys/kernel/tracing/instances/stage56
trace_marker="$trace_root/trace_marker"
benchmark_log=/dev/shm/stage56_trace_soak.log
benchmark_json=/dev/shm/stage56_trace_soak.json
thermal_log=/dev/shm/stage56_trace_soak_thermal.tsv
trace_archive="$stage_root/osnoise/baseline_trace.dat.gz"
comm_name=$(basename "$binary")
comm_name=${comm_name:0:15}
marker_mode=0
tracefs_mode=0
instances_mode=0
instance_mode=0
monitor_pid=

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
export Y26_STAGE53_SPIN_POOL=1
export Y26_STAGE55_FRAME_GATED_SPIN=1

cleanup() {
    if [[ -n $monitor_pid ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if sudo -n test -d "$trace_root"; then
        printf '0\n' | sudo -n tee "$trace_root/tracing_on" >/dev/null || true
    fi
    if [[ $marker_mode != 0 ]] && sudo -n test -e "$trace_marker"; then
        sudo -n chmod "$marker_mode" "$trace_marker" || true
    fi
    if [[ $instance_mode != 0 ]] && sudo -n test -d "$trace_root"; then
        sudo -n chmod "$instance_mode" "$trace_root" || true
    fi
    sudo -n rmdir "$trace_root" 2>/dev/null || true
    [[ $instances_mode == 0 ]] || sudo -n chmod "$instances_mode" /sys/kernel/tracing/instances || true
    [[ $tracefs_mode == 0 ]] || sudo -n chmod "$tracefs_mode" /sys/kernel/tracing || true
}
trap cleanup EXIT

rm -f "$benchmark_log" "$benchmark_json" "$thermal_log"
mkdir -p "$stage_root/osnoise" "$stage_root/profiles"
sudo -n mkdir -p "$trace_root"
tracefs_mode=$(sudo -n stat -c %a /sys/kernel/tracing)
instances_mode=$(sudo -n stat -c %a /sys/kernel/tracing/instances)
instance_mode=$(sudo -n stat -c %a "$trace_root")
marker_mode=$(sudo -n stat -c %a "$trace_marker")
sudo -n chmod o+x /sys/kernel/tracing /sys/kernel/tracing/instances "$trace_root"
printf '0\n' | sudo -n tee "$trace_root/tracing_on" >/dev/null
printf 'mono_raw\n' | sudo -n tee "$trace_root/trace_clock" >/dev/null
printf '65536\n' | sudo -n tee "$trace_root/buffer_size_kb" >/dev/null
printf '1f\n' | sudo -n tee "$trace_root/tracing_cpumask" >/dev/null
printf '\n' | sudo -n tee "$trace_root/trace" >/dev/null

for event in \
    irq/irq_handler_entry irq/irq_handler_exit \
    irq/softirq_entry irq/softirq_exit \
    workqueue/workqueue_execute_start workqueue/workqueue_execute_end \
    block/block_rq_issue block/block_rq_complete \
    power/cpu_frequency; do
    [[ -e $trace_root/events/$event/enable ]] || continue
    printf '1\n' | sudo -n tee "$trace_root/events/$event/enable" >/dev/null
done
if [[ -e $trace_root/events/sched/sched_switch/filter ]]; then
    printf 'prev_comm == "%s" || next_comm == "%s"\n' "$comm_name" "$comm_name" |
        sudo -n tee "$trace_root/events/sched/sched_switch/filter" >/dev/null
    printf '1\n' | sudo -n tee "$trace_root/events/sched/sched_switch/enable" >/dev/null
fi
sudo -n chmod 222 "$trace_marker"

taskset -c 7 bash -c '
    while :; do
        printf "%s" "$(date -u +%FT%TZ)" >>/dev/shm/stage56_trace_soak_thermal.tsv
        for zone in /sys/class/thermal/thermal_zone*/temp; do
            printf "\t%s" "$(cat "$zone")" >>/dev/shm/stage56_trace_soak_thermal.tsv
        done
        for cpu in 0 1 2 3 4; do
            printf "\t%s" "$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq 2>/dev/null || printf unavailable)" >>/dev/shm/stage56_trace_soak_thermal.tsv
        done
        printf "\n" >>/dev/shm/stage56_trace_soak_thermal.tsv
        sleep 5
    done
' &
monitor_pid=$!

printf '1\n' | sudo -n tee "$trace_root/tracing_on" >/dev/null
set +e
Y26_STAGE56_TRACE_MARKER="$trace_marker" timeout 2400 taskset -c 0-4 "$binary" \
    --package "$package" --image "$fixture" --input-mode preprocessed-f32 \
    --output-json "$benchmark_json" --threads 4 --pin 0-3 --scheduler safe \
    --warmup 20 --runs "$runs" --repeats 1 --verify --benchmark \
    >"$benchmark_log" 2>&1
run_status=$?
set -e
printf '0\n' | sudo -n tee "$trace_root/tracing_on" >/dev/null
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

sudo -n cat "$trace_root/trace" | gzip -1 >"$trace_archive"
if sudo -n test -r "$trace_root/trace_stat"; then
    sudo -n cat "$trace_root/trace_stat" >"$stage_root/osnoise/baseline_trace_stat.txt"
else
    printf 'trace_stat unavailable on this kernel\n' \
        >"$stage_root/osnoise/baseline_trace_stat.txt"
fi
cp "$benchmark_log" "$stage_root/osnoise/baseline_trace_soak.log"
cp "$benchmark_json" "$stage_root/osnoise/baseline_trace_soak.json"
cp "$thermal_log" "$stage_root/osnoise/baseline_trace_soak_thermal.tsv"
sha256sum "$trace_archive" "$stage_root/osnoise/baseline_trace_soak.log" \
    "$stage_root/osnoise/baseline_trace_soak.json"
exit "$run_status"
