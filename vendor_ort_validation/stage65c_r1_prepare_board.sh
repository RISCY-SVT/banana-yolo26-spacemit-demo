#!/usr/bin/env bash
set -euo pipefail

STAGE_ID=${STAGE_ID:-BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001}
BOARD=${BOARD:-svt@banana}
BOARD_ROOT="/data/k1x-stage-runs/${STAGE_ID}"
SOURCE_STAGE=/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65C-A1-VS-B2-K1X-SPACEMIT-EP-PLACEMENT-CORRECTNESS-COCO-PERFORMANCE-AND-STABILITY-GATE-001
HOST_BUILD=${HOST_BUILD:?HOST_BUILD is required}

ssh -o BatchMode=yes "$BOARD" bash -s -- "$BOARD_ROOT" "$SOURCE_STAGE" "$STAGE_ID" <<'REMOTE'
set -euo pipefail
root=$1
source_root=$2
stage_id=$3
source_device=$(findmnt -n -o SOURCE --target /data)
case "$source_device" in
  /dev/mmc*) printf 'refusing eMMC-backed /data: %s\n' "$source_device" >&2; exit 2 ;;
esac
[[ -d $source_root ]] || { printf 'missing accepted Stage65C board root\n' >&2; exit 2; }
if [[ -e $root ]] && find "$root" -mindepth 1 -print -quit | grep -q .; then
  printf 'board R1 root is not empty: %s\n' "$root" >&2
  exit 2
fi
mkdir -p "$root"/{bin,models,runtime,opencv,h500,manifests,state,diagnostic-inputs,boundaries,determinism,coco,tmp,cache}
cp -a "$source_root/runtime/." "$root/runtime/"
cp -a "$source_root/opencv/." "$root/opencv/"
cp -a "$source_root/bin/stage64_two_stage_runner" "$root/bin/"
cp -a "$source_root/bin/stage64_two_stage_coco" "$root/bin/"
cp -a "$source_root/bin/stage64_run_two_stage_board.sh" "$root/bin/"
cp -a "$source_root/bin/stage64_run_coco_board.sh" "$root/bin/"
cp -a "$source_root/models/stage65b_r1_a1.inference.onnx" "$root/models/"
cp -a "$source_root/models/stage65b_r1_b2.inference.onnx" "$root/models/"
cp -a "$source_root/models/stage65b_r1_a1.postprocess.onnx" "$root/models/"
cp -a "$source_root/h500/images" "$root/h500/"
cp -a "$source_root/manifests/selection_H500_holdout.txt" "$root/manifests/"

{
  printf 'field\tvalue\n'
  printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'boot_id\t%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
  printf 'kernel\t%s\n' "$(uname -a)"
  printf 'device_model\t%s\n' "$(tr -d '\000' </proc/device-tree/model 2>/dev/null || true)"
  printf 'device_serial\t%s\n' "$(tr -d '\000' </proc/device-tree/serial-number 2>/dev/null || true)"
  printf 'allowed_cpu_list\t%s\n' "$(cat /proc/self/status | awk '/Cpus_allowed_list/ {print $2}')"
  printf 'data_mount\t%s\n' "$(findmnt -T /data -no TARGET,SOURCE,FSTYPE,OPTIONS)"
  printf 'root_mount\t%s\n' "$(findmnt -T / -no TARGET,SOURCE,FSTYPE,OPTIONS)"
  printf 'data_free_bytes\t%s\n' "$(df -PB1 /data | awk 'NR==2 {print $4}')"
  find /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor -type f -print0 2>/dev/null | sort -z | xargs -0 -r -n1 sh -c 'printf "governor\t%s=%s\n" "$1" "$(cat "$1")"' sh
  find /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq -type f -print0 2>/dev/null | sort -z | xargs -0 -r -n1 sh -c 'printf "frequency_khz\t%s=%s\n" "$1" "$(cat "$1")"' sh
  find /sys/class/thermal/thermal_zone*/temp -type f -print0 2>/dev/null | sort -z | xargs -0 -r -n1 sh -c 'printf "temperature_millic\t%s=%s\n" "$1" "$(cat "$1")"' sh
} >"$root/state/system_state_before.tsv"

{
  printf 'surface\tvalue\n'
  printf 'stage_root_mount\t%s\n' "$(findmnt -T "$root" -no SOURCE,FSTYPE,OPTIONS)"
  printf 'root_mount\t%s\n' "$(findmnt -T / -no SOURCE,FSTYPE,OPTIONS)"
  printf 'emmc_stage_path_count\t%s\n' "$(find / -xdev -path "*$stage_id*" -print 2>/dev/null | wc -l)"
} >"$root/state/storage_write_audit_before.tsv"
REMOTE

rsync -a "$HOST_BUILD/stage65c_r1_tail_replay" "$BOARD:$BOARD_ROOT/bin/"
rsync -a \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage65c_r1_board_boundaries.sh \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage65c_r1_board_determinism.sh \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage65c_r1_board_finalize.sh \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage65c_board_coco.sh \
  "$BOARD:$BOARD_ROOT/bin/"

ssh -o BatchMode=yes "$BOARD" bash -s -- "$BOARD_ROOT" <<'REMOTE'
set -euo pipefail
root=$1
chmod +x "$root"/bin/*
{
  sha256sum "$root/models/stage65b_r1_a1.inference.onnx"
  sha256sum "$root/models/stage65b_r1_b2.inference.onnx"
  sha256sum "$root/models/stage65b_r1_a1.postprocess.onnx"
  sha256sum "$root/runtime/lib/libonnxruntime.so.1.24.2+spacemit.a1"
  sha256sum "$root/runtime/lib/libspacemit_ep.so.2.0.6"
  sha256sum "$root/bin/stage64_two_stage_runner"
  sha256sum "$root/bin/stage64_two_stage_coco"
  sha256sum "$root/bin/stage65c_r1_tail_replay"
} >"$root/manifests/board_input_sha256.txt"
find "$root/h500/images" -maxdepth 1 -type f | wc -l >"$root/manifests/h500_image_count.txt"
REMOTE

printf 'prepared %s on %s\n' "$BOARD_ROOT" "$BOARD"
