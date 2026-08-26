#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65E_BOARD_ROOT:?STAGE65E_BOARD_ROOT is required}"
root=$STAGE65E_BOARD_ROOT
out="$root/file-pipeline-stage65e"
images="$root/h500/images"
runner="$root/bin/stage64_two_stage_coco"
runtime="$root/runtime/lib"
opencv="$root/opencv/lib"
tail="$root/models/stage65b_r1_b2.postprocess.onnx"
ulimit -c 0

[[ ! -e $out ]] || { printf 'file-pipeline output exists: %s\n' "$out" >&2; exit 2; }
image_count=$(find "$images" -maxdepth 1 -type f -name '*.jpg' | wc -l)
[[ $image_count -eq 500 ]] || {
  printf 'H500 image count mismatch: %s != 500\n' "$image_count" >&2
  exit 2
}
mkdir -p "$out"
printf 'surface\timages\texit_code\tprediction_sha256\ttiming_sha256\n' >"$out/status.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

for model in B2 C2; do
  model_out="$out/$model"
  rc=0
  set +e
  timeout --signal=TERM --kill-after=30s 3600s \
    "$root/bin/stage65d_run_coco_board.sh" \
      --runner "$runner" \
      --runtime-lib "$runtime" \
      --opencv-lib "$opencv" \
      --inference "$(inference_for "$model")" \
      --tail "$tail" \
      --images "$images" \
      --output-dir "$model_out" \
      --provider spacemit \
      --cpu-list 0-3 \
      --threads 4 \
      --confidence 0.001 \
      --limit 100
  rc=$?
  set -e
  prediction_sha=missing
  timing_sha=missing
  [[ -f $model_out/predictions.json ]] && prediction_sha=$(sha256sum "$model_out/predictions.json" | awk '{print $1}')
  [[ -f $model_out/timing.tsv ]] && timing_sha=$(sha256sum "$model_out/timing.tsv" | awk '{print $1}')
  printf '%s\t100\t%s\t%s\t%s\n' "$model" "$rc" "$prediction_sha" "$timing_sha" \
    >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || exit "$rc"
  [[ $(($(wc -l <"$model_out/timing.tsv") - 1)) -eq 100 ]] || {
    printf 'timing row count mismatch: %s\n' "$model_out/timing.tsv" >&2
    exit 3
  }
done

env LD_LIBRARY_PATH="$opencv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  python3 "$root/bin/stage65d_r1_file_pipeline.py" \
    --images "$images" \
    --output "$out/file-components.tsv" \
    --limit 100 \
    --repeats 3
[[ $(($(wc -l <"$out/file-components.tsv") - 1)) -eq 300 ]] || {
  printf 'file-component row count mismatch\n' >&2
  exit 3
}
sha256sum "$out/status.raw.tsv" "$out/file-components.tsv" >"$out/evidence-sha256.txt"
printf 'stage65e_board_file_pipeline status=pass\n'
