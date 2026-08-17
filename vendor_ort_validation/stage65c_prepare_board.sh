#!/usr/bin/env bash
set -euo pipefail

STAGE_ID=${STAGE_ID:-BANANA-YOLO26-XSLIM-STAGE65C-A1-VS-B2-K1X-SPACEMIT-EP-PLACEMENT-CORRECTNESS-COCO-PERFORMANCE-AND-STABILITY-GATE-001}
BOARD=${BOARD:-svt@banana}
BOARD_ROOT="/data/k1x-stage-runs/${STAGE_ID}"
DEV_ROOT=/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001
R1_ROOT=/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65B-R1-COCO-TRAIN2017-EVALUATION-DISJOINT-CORPUS-PTQ-GRAPHWISE-AND-PYRAMID-CAUSAL-LOCALIZATION-001
S64_ROOT=/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001
S63_BOARD=/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE63-RT206-SPACEMIT-EP-INT8-REGRESSION-REVALIDATION-PLUGIN-FULLMODEL-COCO-AND-ISSUE1-GATE-001
S64_BOARD=/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE64-XSLIM211-AND-VENDOR-COMMIT-S8-QDQ-YOLO26-FULLMODEL-COCO-AND-RT206-GATE-001
RUNTIME=/data/vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6
DATASET=/data/datasets/coco2017-independent-stage65b-r1

ssh -o BatchMode=yes "$BOARD" bash -s -- "$BOARD_ROOT" "$S63_BOARD" "$S64_BOARD" <<'REMOTE'
set -euo pipefail
root=$1
s63=$2
s64=$3
source_device=$(findmnt -n -o SOURCE --target /data)
case "$source_device" in
  /dev/mmc*) printf 'refusing eMMC /data: %s\n' "$source_device" >&2; exit 2 ;;
esac
if [[ -e $root ]] && find "$root" -mindepth 1 -print -quit | grep -q .; then
  printf 'board Stage root is not empty: %s\n' "$root" >&2
  exit 2
fi
mkdir -p "$root"/{bin,models,manifests,runtime,opencv,fixtures/tiny,fixtures/fixed,tiny-controls,plugin,profile,h500/images,coco,performance,soak,tmp,cache,state,vendor-control/runners,vendor-control/runtimes/rt206}
cp -a "$s63/opencv/lib" "$root/opencv/"
cp -a "$s63/runners/runner_rt206" "$root/bin/vendor_single_model_runner"
cp -a "$s63/runners/runner_rt206" "$root/vendor-control/runners/runner_rt206"
cp -a "$s64/models/tiny/." "$root/models/tiny/"
cp -a "$s64/fixtures/tiny/." "$root/fixtures/tiny/"
cp -a "$s64/fixtures/images_F0_f32.bin" "$root/fixtures/fixed/"
cp -a "$s63/plugin/." "$root/plugin/"
REMOTE

rsync -a --delete "$RUNTIME/" "$BOARD:$BOARD_ROOT/runtime/"
rsync -a \
  "$S64_ROOT/bin/stage64_two_stage_runner" \
  "$S64_ROOT/bin/stage64_two_stage_coco" \
  "$S64_ROOT/host/build/public-repro-runner/stage64_single_model_runner" \
  "$BOARD:$BOARD_ROOT/bin/"
rsync -a \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage64_run_tiny_board.sh \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage64_run_two_stage_board.sh \
  /data/worktrees/banana-yolo26-xslim211-s8-qdq-validation/vendor_ort_validation/stage64_run_coco_board.sh \
  "$BOARD:$BOARD_ROOT/bin/"
rsync -a \
  "$DEV_ROOT/candidates/quantization/A1/run1/output/xslim_dev_001a_a1_split_s8_qdq.onnx" \
  "$DEV_ROOT/candidates/postprocess/A1/models/stage65b_r1_a1.inference.onnx" \
  "$DEV_ROOT/candidates/postprocess/A1/models/stage65b_r1_a1.postprocess.onnx" \
  "$R1_ROOT/quantization/B2/run1/output/stage65b_r1_b2_split_s8_qdq.onnx" \
  "$R1_ROOT/postprocess/B2/candidate-gate/B2/models/stage65b_r1_b2.inference.onnx" \
  "$BOARD:$BOARD_ROOT/models/"
rsync -a \
  "$DEV_ROOT/candidates/quantization/A1/run1/range-policy-manifest.json" \
  "$R1_ROOT/dataset/selection_H500_holdout.txt" \
  "$BOARD:$BOARD_ROOT/manifests/"
rsync -a \
  "$S64_ROOT/host/accepted-preprocessed/bus_r640_nchw_f32.bin" \
  "$S64_ROOT/host/accepted-preprocessed/canonical_640_nchw_f32.bin" \
  "$S64_ROOT/host/accepted-preprocessed/zidane_r640_nchw_f32.bin" \
  "$BOARD:$BOARD_ROOT/fixtures/fixed/"

rsync -a --files-from="$R1_ROOT/dataset/selection_H500_holdout.txt" \
  "$DATASET/selected/images/" "$BOARD:$BOARD_ROOT/h500/images/"

ssh -o BatchMode=yes "$BOARD" bash -s -- "$BOARD_ROOT" <<'REMOTE'
set -euo pipefail
root=$1
chmod +x "$root"/bin/*
ln -sfn ../../../runtime/lib "$root/vendor-control/runtimes/rt206/lib"
find "$root/h500/images" -maxdepth 1 -type f | wc -l >"$root/manifests/h500_image_count.txt"
{
  sha256sum "$root/models/xslim_dev_001a_a1_split_s8_qdq.onnx"
  sha256sum "$root/models/stage65b_r1_a1.inference.onnx"
  sha256sum "$root/models/stage65b_r1_a1.postprocess.onnx"
  sha256sum "$root/models/stage65b_r1_b2_split_s8_qdq.onnx"
  sha256sum "$root/models/stage65b_r1_b2.inference.onnx"
  sha256sum "$root/runtime/lib/libonnxruntime.so.1.24.2+spacemit.a1"
  sha256sum "$root/runtime/lib/libspacemit_ep.so.2.0.6"
  sha256sum "$root/bin/stage64_two_stage_runner"
  sha256sum "$root/bin/stage64_two_stage_coco"
  sha256sum "$root/bin/stage64_single_model_runner"
  sha256sum "$root/bin/vendor_single_model_runner"
} >"$root/manifests/board_input_sha256.txt"
find "$root" -xdev -type f -printf '%P\t%s\n' | sort >"$root/manifests/board_input_files.tsv"
findmnt -T "$root" -o TARGET,SOURCE,FSTYPE,OPTIONS >"$root/manifests/storage_mount.txt"
REMOTE

printf 'prepared %s on %s\n' "$BOARD_ROOT" "$BOARD"
