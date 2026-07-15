#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <stage-root>\n' "$0" >&2
    exit 2
fi

destination="$1/candidate-builds/env"
mkdir -p "$destination"

write_base() {
    cat <<'EOF'
unset Y26_STAGE56_DENSE_FUSED_M8
unset Y26_STAGE56_STEM_K32
unset Y26_STAGE56_LUT2_FACTOR
unset Y26_STAGE56_ATTENTION_DIRECT_PACK
unset Y26_STAGE56_HEAD_PRODUCER_REDUCTION
unset Y26_STAGE56_MMAP_ARENA
unset Y26_STAGE56_PREFAULT
unset Y26_STAGE56_MLOCK
unset Y26_STAGE56_MADV_HUGEPAGE
unset Y26_STAGE56_MADV_COLLAPSE
unset Y26_STAGE54_DENSE_M8
unset Y26_STAGE54_DENSE_WEIGHT_STATIONARY
unset Y26_STAGE54_DENSE_PARTITION
unset Y26_STAGE54_INPUT_STEM_FUSED
unset Y26_STAGE55_DENSE_FAMILY_B
unset Y26_STAGE55_DEPTHWISE_E2C4
EOF
}

write_arm() {
    local name=$1
    shift
    {
        write_base
        printf '%s\n' "$@"
    } >"$destination/$name.env"
}

write_arm control ':'
write_arm dense_fused_m8 'export Y26_STAGE56_DENSE_FUSED_M8=1'
write_arm stem_k32 'export Y26_STAGE56_STEM_K32=1'
write_arm lut2_factor 'export Y26_STAGE56_LUT2_FACTOR=1'
write_arm attention_direct_pack 'export Y26_STAGE56_ATTENTION_DIRECT_PACK=1'
write_arm head_producer 'export Y26_STAGE56_HEAD_PRODUCER_REDUCTION=1'
write_arm head_attention \
    'export Y26_STAGE56_HEAD_PRODUCER_REDUCTION=1' \
    'export Y26_STAGE56_ATTENTION_DIRECT_PACK=1'
write_arm depthwise_e2c4 'export Y26_STAGE55_DEPTHWISE_E2C4=1'
write_arm rectangular_m8 'export Y26_STAGE54_DENSE_M8=1'
write_arm rectangular_weight_stationary 'export Y26_STAGE54_DENSE_WEIGHT_STATIONARY=1'
write_arm rectangular_family_b 'export Y26_STAGE55_DENSE_FAMILY_B=1'
write_arm rectangular_partition_n 'export Y26_STAGE54_DENSE_PARTITION=1'
write_arm rectangular_partition_2d 'export Y26_STAGE54_DENSE_PARTITION=2'

sha256sum "$destination"/*.env >"$destination/sha256.txt"
