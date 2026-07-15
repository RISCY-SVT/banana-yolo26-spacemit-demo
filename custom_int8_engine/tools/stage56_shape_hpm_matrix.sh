#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s <stage-root> <shape-binary> <package-manifest-sha256>\n' "$0" >&2
    exit 2
fi

stage_root=$1
shape_binary=$2
manifest_sha256=$3

export Y26_STAGE56_SHAPE_OPERATIONS=5,12,124,133,144,161,201,206
export Y26_STAGE56_NATIVE_SHAPES_ONLY=1
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

events=(
    instructions
    frontend_stalls
    backend_stalls
    l1d_read_access
    l1d_read_miss
    l1i_read_access
    l1i_read_miss
    dtlb_read_miss
    dtlb_write_miss
    itlb_read_miss
)

mkdir -p "$stage_root/hpm"
for event in "${events[@]}"; do
    export Y26_STAGE55_PERF_GROUP="cycles,$event"
    log="/dev/shm/stage56_shape_hpm_$event.log"
    taskset -c 0-4 "$shape_binary" \
        "$stage_root/packages/stage55-control" "$manifest_sha256" 2 5 \
        >"$log" 2>&1
    cp "$log" "$stage_root/hpm/shapes_$event.log"
    printf '[%s] rows=%s counters=%s\n' "$event" \
        "$(grep -c '^640' "$log")" \
        "$(grep -c '^stage56_shape_counter' "$log")"
done
