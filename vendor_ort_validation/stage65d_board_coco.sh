#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s h500|val B2|C2 cpu|spacemit\n' "$0" >&2
}

[[ $# -eq 3 ]] || { usage; exit 2; }
: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
dataset=$1
model=$2
provider=$3
root=$STAGE65D_BOARD_ROOT
ulimit -c 0

case "$dataset" in
  h500) images="$root/h500/images"; expected=500 ;;
  val)
    images=/data/k1x-stage-runs/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE46-RT205-SPACEMIT-EP-INT8-FULL-REVALIDATION-PLUGIN-COCO-GATE-001/datasets/coco-val2017/val2017
    expected=5000
    ;;
  *) usage; exit 2 ;;
esac
case "$model" in
  B2) inference="$root/models/stage65b_r1_b2.inference.onnx" ;;
  C2) inference="$root/models/c2_t6_rank_qp.inference.onnx" ;;
  *) usage; exit 2 ;;
esac
[[ $provider == cpu || $provider == spacemit ]] || { usage; exit 2; }

actual=$(find "$images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
[[ $actual -eq $expected ]] || {
  printf 'image count mismatch: %s != %s in %s\n' "$actual" "$expected" "$images" >&2
  exit 2
}

output="$root/coco/$dataset/${model}-${provider}"
status_file="$root/coco/$dataset/status.raw.tsv"
mkdir -p "$(dirname "$status_file")"
if [[ ! -f $status_file ]]; then
  printf 'dataset\tmodel\tprovider\timages\texit_code\tprediction_sha256\ttiming_sha256\n' >"$status_file"
fi
[[ ! -e $output ]] || {
  printf 'refusing existing output: %s\n' "$output" >&2
  exit 2
}

unset SPACEMIT_EP_DUMP_SUBGRAPHS SPACEMIT_EP_DUMP_SUBGRAPHS_DIR
rc=0
set +e
timeout --signal=TERM --kill-after=30s 21600s \
  "$root/bin/stage65d_run_coco_board.sh" \
    --runner "$root/bin/stage64_two_stage_coco" \
    --runtime-lib "$root/runtime/lib" \
    --opencv-lib "$root/opencv/lib" \
    --inference "$inference" \
    --tail "$root/models/stage65b_r1_b2.postprocess.onnx" \
    --images "$images" \
    --output-dir "$output" \
    --provider "$provider" \
    --cpu-list 0-3 \
    --threads 4 \
    --confidence 0.001 \
    --limit 0
rc=$?
set -e

prediction_hash=missing
timing_hash=missing
[[ -f $output/predictions.json ]] && prediction_hash=$(sha256sum "$output/predictions.json" | awk '{print $1}')
[[ -f $output/timing.tsv ]] && timing_hash=$(sha256sum "$output/timing.tsv" | awk '{print $1}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$dataset" "$model" "$provider" "$actual" "$rc" "$prediction_hash" "$timing_hash" \
  >>"$status_file"
exit "$rc"
