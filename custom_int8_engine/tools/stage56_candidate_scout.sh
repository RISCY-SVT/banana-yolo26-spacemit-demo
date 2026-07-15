#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    printf 'usage: %s <stage-root> <binary> <env-file> [env-file ...]\n' "$0" >&2
    exit 2
fi

stage_root=$1
binary=$2
shift 2
package="$stage_root/packages/stage55-control"
fixture="$stage_root/fixtures/full_f0_f7/F0_nchw_f32.bin"
destination="$stage_root/candidate-builds/scouts"
ram_root=/dev/shm/y26-stage56-scouts
mkdir -p "$destination" "$ram_root"

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
unset Y26_STAGE54_HEAD_V2 || true

printf 'candidate\texit_code\traw_samples\toutput_hash\tbinary_sha256\tpackage_sha256\n' \
    >"$destination/scout_matrix.tsv"
overall=0
for env_file in "$@"; do
    [[ -f $env_file ]] || { printf 'missing environment: %s\n' "$env_file" >&2; exit 2; }
    name=${env_file##*/}
    name=${name%.env}
    # shellcheck disable=SC1090
    source "$env_file"
    log="$ram_root/$name.log"
    output="$ram_root/$name.json"
    rm -f "$log" "$output"
    set +e
    taskset -c 0-4 "$binary" \
        --package "$package" --image "$fixture" --input-mode preprocessed-f32 \
        --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
        --warmup 5 --runs 20 --repeats 1 --verify --benchmark >"$log" 2>&1
    status=$?
    set -e
    samples=$(grep -c '^raw' "$log" || true)
    output_hash=$(awk -F'\t' '/^raw/ {for (i=1; i<=NF; ++i) if ($i ~ /^hash=/) hash=substr($i,6)} END {print hash}' "$log")
    cp "$log" "$destination/$name.log"
    [[ -f $output ]] && cp "$output" "$destination/$name.json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$samples" \
        "${output_hash:-unavailable}" "$(sha256sum "$binary" | awk '{print $1}')" \
        "$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')" \
        >>"$destination/scout_matrix.tsv"
    [[ $status == 0 ]] || overall=1
done
sha256sum "$destination"/* >"$destination/sha256.txt"
exit "$overall"
