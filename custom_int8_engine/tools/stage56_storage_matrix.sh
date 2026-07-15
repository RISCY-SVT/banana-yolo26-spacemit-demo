#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    printf 'usage: %s <stage-root> <binary> <package> <fixture>\n' "$0" >&2
    exit 2
fi

stage_root=$1
source_binary=$2
source_package=$3
source_fixture=$4
source_opencv="$stage_root/opencv/lib"
tmpfs_root=/dev/shm/y26-stage56-storage-runtime
emmc_root=/home/svt/.cache/y26-stage56-storage-runtime
matrix_root="$stage_root/storage"
helper="$stage_root/bin/stage56_benchmark_arm.sh"

cleanup() {
    rm -rf "$tmpfs_root" "$emmc_root"
}
trap cleanup EXIT

for path in "$source_binary" "$source_fixture" "$helper"; do
    [[ -f $path ]] || { printf 'missing input: %s\n' "$path" >&2; exit 2; }
done
[[ -d $source_package ]] || { printf 'missing package: %s\n' "$source_package" >&2; exit 2; }
[[ -d $source_opencv ]] || { printf 'missing OpenCV runtime: %s\n' "$source_opencv" >&2; exit 2; }
rm -rf "$tmpfs_root" "$emmc_root"
mkdir -p "$matrix_root" "$tmpfs_root" "$emmc_root"

copy_runtime() {
    local root=$1
    mkdir -p "$root/bin" "$root/packages" "$root/fixtures" "$root/opencv"
    cp "$source_binary" "$root/bin/yolo26_k1x_int8"
    cp -a "$source_package" "$root/packages/model"
    cp "$source_fixture" "$root/fixtures/F0_nchw_f32.bin"
    cp -a "$source_opencv" "$root/opencv/lib"
    chmod 0755 "$root/bin/yolo26_k1x_int8"
}

copy_runtime "$tmpfs_root"
copy_runtime "$emmc_root"
sync "$emmc_root"

{
    printf 'location\tbytes\tfiles\tbinary_sha256\tfixture_sha256\tpackage_manifest_sha256\topencv_tree_sha256\n'
    for location in nvme tmpfs emmc; do
        case "$location" in
            nvme) root=source ;;
            tmpfs) root=$tmpfs_root ;;
            emmc) root=$emmc_root ;;
        esac
        if [[ $root == source ]]; then
            binary=$source_binary
            package=$source_package
            fixture=$source_fixture
            opencv=$source_opencv
            bytes=$(du -sb "$source_binary" "$source_package" "$source_fixture" "$source_opencv" | awk '{sum += $1} END {print sum}')
            files=$((1 + $(find "$source_package" -type f | wc -l) + 1 + $(find "$source_opencv" -type f | wc -l)))
        else
            binary="$root/bin/yolo26_k1x_int8"
            package="$root/packages/model"
            fixture="$root/fixtures/F0_nchw_f32.bin"
            opencv="$root/opencv/lib"
            bytes=$(du -sb "$root" | awk '{print $1}')
            files=$(find "$root" -type f | wc -l)
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$location" "$bytes" "$files" \
            "$(sha256sum "$binary" | awk '{print $1}')" \
            "$(sha256sum "$fixture" | awk '{print $1}')" \
            "$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')" \
            "$(cd "$opencv" && find . -type f -printf '%P\n' | LC_ALL=C sort | while read -r file; do printf '%s  %s\n' "$(sha256sum "$file" | awk '{print $1}')" "$file"; done | sha256sum | awk '{print $1}')"
    done
} >"$matrix_root/runtime_identity.tsv"

run_location() {
    local location=$1
    local binary package fixture
    case "$location" in
        nvme)
            binary=$source_binary; package=$source_package; fixture=$source_fixture ;;
        tmpfs)
            binary="$tmpfs_root/bin/yolo26_k1x_int8"
            package="$tmpfs_root/packages/model"
            fixture="$tmpfs_root/fixtures/F0_nchw_f32.bin" ;;
        emmc)
            binary="$emmc_root/bin/yolo26_k1x_int8"
            package="$emmc_root/packages/model"
            fixture="$emmc_root/fixtures/F0_nchw_f32.bin" ;;
        *) return 2 ;;
    esac

    local cold_log="/dev/shm/stage56_storage_${location}_cold.log"
    local cold_time="/dev/shm/stage56_storage_${location}_cold.time"
    local cold_json="/dev/shm/stage56_storage_${location}_cold.json"
    rm -f "$cold_log" "$cold_time" "$cold_json"
    taskset -c 0-4 /usr/bin/time -v "$binary" \
        --package "$package" --image "$fixture" --input-mode preprocessed-f32 \
        --output-json "$cold_json" --threads 4 --pin 0-3 --scheduler safe \
        --warmup 0 --runs 1 --repeats 1 --verify --benchmark \
        >"$cold_log" 2>"$cold_time"
    cp "$cold_log" "$cold_time" "$cold_json" "$matrix_root/"

    Y26_STAGE56_BINARY="$binary" Y26_STAGE56_PACKAGE="$package" \
    Y26_STAGE56_FIXTURE="$fixture" \
        "$helper" "$stage_root" "storage_${location}_warm" low-latency 100 5
}

run_location nvme
run_location tmpfs
run_location emmc

cp "$matrix_root/runtime_identity.tsv" "$matrix_root/emmc_test_manifest.tsv"
printf 'removed_after_test\t%s\n' "$emmc_root" >>"$matrix_root/emmc_test_manifest.tsv"
