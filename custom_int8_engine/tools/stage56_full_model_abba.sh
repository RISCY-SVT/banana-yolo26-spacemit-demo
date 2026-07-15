#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 10 ]]; then
    printf 'usage: %s <stage-root> <label> <binary> <package> <fixture> <a-env> <b-env> <runs> [cycles] [warmup]\n' "$0" >&2
    exit 2
fi

stage_root=$1
label=$2
binary=$3
package=$4
fixture=$5
a_env=$6
b_env=$7
runs=$8
cycles=${9:-5}
warmup=${10:-10}

case "$label" in
    *[!A-Za-z0-9_.-]*) printf 'invalid label: %s\n' "$label" >&2; exit 2 ;;
esac
for path in "$binary" "$fixture" "$a_env" "$b_env"; do
    [[ -f $path ]] || { printf 'missing input: %s\n' "$path" >&2; exit 2; }
done
[[ -d $package ]] || { printf 'missing package: %s\n' "$package" >&2; exit 2; }

ram_dir="/dev/shm/y26-stage56-abba-$label"
destination="$stage_root/profiles/$label"
rm -rf "$ram_dir"
mkdir -p "$ram_dir" "$destination"
combined="$ram_dir/combined.log"
blocks="$ram_dir/blocks.tsv"
printf 'block\tcycle\tposition\tarm\tenv_file\tlog\toutput_json\texit_code\n' >"$blocks"
: >"$combined"

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
unset Y26_STAGE54_HEAD_V2 Y26_STAGE55_DEPTHWISE_E2C4 || true

block=0
run_status=0
for ((cycle = 0; cycle < cycles; ++cycle)); do
    if ((cycle % 2 == 0)); then
        order=(A B)
    else
        order=(B A)
    fi
    for position in 0 1; do
        arm=${order[$position]}
        if [[ $arm == A ]]; then
            env_file=$a_env
        else
            env_file=$b_env
        fi
        # Each env file must explicitly unset candidate variables it does not use.
        # shellcheck disable=SC1090
        source "$env_file"
        printf 'block\t%s\t%s\t%s\t%s\n' "$cycle" "$position" "$arm" "$block" \
            >>"$combined"
        block_log="$ram_dir/$(printf '%03d' "$block")_${arm}.log"
        output_json="$ram_dir/$(printf '%03d' "$block")_${arm}.json"
        set +e
        taskset -c 0-4 "$binary" \
            --package "$package" --image "$fixture" --input-mode preprocessed-f32 \
            --output-json "$output_json" --threads 4 --pin 0-3 --scheduler safe \
            --warmup "$warmup" --runs "$runs" --repeats 1 --verify --benchmark \
            >"$block_log" 2>&1
        status=$?
        set -e
        cat "$block_log" >>"$combined"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$block" "$cycle" "$position" "$arm" "$env_file" "$block_log" \
            "$output_json" "$status" >>"$blocks"
        ((block += 1))
        if [[ $status != 0 ]]; then
            run_status=$status
            break 2
        fi
    done
done

cp -a "$ram_dir/." "$destination/"
{
    sha256sum "$binary" "$fixture" "$a_env" "$b_env"
    sha256sum "$package/asset_hashes.tsv"
    find "$destination" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum
} >"$destination/sha256.txt"
exit "$run_status"
