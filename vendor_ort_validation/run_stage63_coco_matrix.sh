#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ($1 != smoke && $1 != full) ]]; then
    echo "usage: $0 smoke|full" >&2
    exit 2
fi

: "${STAGE63_BOARD_ROOT:?set STAGE63_BOARD_ROOT}"
: "${STAGE63_OPENCV_ROOT:?set STAGE63_OPENCV_ROOT}"
: "${STAGE63_COCO_IMAGES:?set STAGE63_COCO_IMAGES}"

mode=$1
root=$STAGE63_BOARD_ROOT
runner="$root/run_coco_surface.sh"
labels="$root/models/coco80.txt"
out="$root/coco/$mode"
mkdir -p "$out"

run_surface() {
    local runtime=$1
    local provider=$2
    local model_surface=$3
    local limit=$4
    local surface="${runtime}_${provider}_${model_surface}_${mode}"
    local predictor="$root/bin/yolo26_coco_predict_${runtime}"
    local model
    case "$model_surface" in
        fp32) model="$root/models/yolo26n_640_e2e_fp32.onnx" ;;
        fp16) model="$root/models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx" ;;
        int8) model="$root/models/manual_e2e_rep_conv_matmul_qdq.onnx" ;;
        *) echo "unsupported model surface: $model_surface" >&2; return 2 ;;
    esac
    [[ -x $predictor ]] || {
        printf 'missing version-bound predictor: %s\n' "$predictor" >&2
        return 2
    }
    "$runner" \
        --predictor "$predictor" \
        --runtime "$root/runtimes/$runtime" \
        --opencv "$STAGE63_OPENCV_ROOT" \
        --model "$model" \
        --labels "$labels" \
        --images "$STAGE63_COCO_IMAGES" \
        --output-dir "$out/$surface" \
        --surface "$surface" \
        --provider "$provider" \
        --limit "$limit"
    printf 'completed surface=%s\n' "$surface"
}

if [[ $mode == smoke ]]; then
    export STAGE63_COCO_TIMEOUT_SECONDS=900
    run_surface rt204 spacemit fp32 10
    run_surface rt204 spacemit fp16 10
    run_surface rt204 cpu int8 10
    run_surface rt205 cpu int8 10
    run_surface rt206 cpu fp32 10
    run_surface rt206 cpu int8 10
    run_surface rt206 spacemit fp32 10
    run_surface rt206 spacemit fp16 10
else
    export STAGE63_COCO_TIMEOUT_SECONDS=14400
    run_surface rt206 spacemit fp16 0
    run_surface rt206 spacemit fp32 0
    run_surface rt206 cpu int8 0
    run_surface rt206 cpu fp32 0
fi
