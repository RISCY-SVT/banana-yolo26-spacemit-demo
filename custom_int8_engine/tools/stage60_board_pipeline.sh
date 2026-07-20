#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STAGE_ROOT IMAGE" >&2
  exit 2
fi

stage_root=$1
image=$2
ram=/dev/shm/y26-stage60-pipeline-$$
output=$stage_root/pipeline
mkdir -p "$ram" "$output"
trap 'rm -rf "$ram"' EXIT

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
export Y26_STAGE56_HEAD_PRODUCER_REDUCTION=1
export Y26_STAGE56_ATTENTION_DIRECT_PACK=1
export Y26_STAGE57_E2C5=1
export Y26_STAGE57_ATTENTION_MATMUL_C8=1
export Y26_STAGE57_RGB_COPY_RVV=1

opencv_lib=${Y26_STAGE60_OPENCV_LIB:-$stage_root/lib/opencv}
[[ -d $opencv_lib ]] || {
  echo "Stage60 OpenCV runtime closure is missing: $opencv_lib" >&2
  exit 1
}
export LD_LIBRARY_PATH=$opencv_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for resolution in 640 512 448 416 384 352 320 256; do
  package=$stage_root/packages/r$resolution
  set +e
  "$stage_root/bin/stage52_image_pipeline_bench" \
    --package "$package" --image "$image" --threads 4 \
    --warmup 10 --runs 100 --repeats 5 \
    >"$ram/r${resolution}_serial.tsv" 2>"$ram/r${resolution}_serial.stderr"
  serial_status=$?
  set -e
  printf '%s\n' "$serial_status" >"$ram/r${resolution}_serial.exit-status.txt"
  cp "$ram/r${resolution}_serial.stderr" "$output/r${resolution}_serial.stderr"
  cp "$ram/r${resolution}_serial.exit-status.txt" \
    "$output/r${resolution}_serial.exit-status.txt"
  if [[ -f $ram/r${resolution}_serial.tsv ]]; then
    cp "$ram/r${resolution}_serial.tsv" "$output/r${resolution}_serial.tsv"
  fi
  [[ $serial_status -eq 0 ]] || {
    echo "serial pipeline failed for R$resolution" >&2
    exit "$serial_status"
  }

  set +e
  "$stage_root/bin/stage56_double_buffer_bench" \
    --package "$package" --image "$image" \
    --warmup 10 --runs 100 --repeats 5 \
    >"$ram/r${resolution}_double_buffer.tsv" \
    2>"$ram/r${resolution}_double_buffer.stderr"
  double_status=$?
  set -e
  printf '%s\n' "$double_status" >"$ram/r${resolution}_double_buffer.exit-status.txt"
  cp "$ram/r${resolution}_double_buffer.stderr" \
    "$output/r${resolution}_double_buffer.stderr"
  cp "$ram/r${resolution}_double_buffer.exit-status.txt" \
    "$output/r${resolution}_double_buffer.exit-status.txt"
  if [[ -f $ram/r${resolution}_double_buffer.tsv ]]; then
    cp "$ram/r${resolution}_double_buffer.tsv" "$output/r${resolution}_double_buffer.tsv"
  fi
  [[ $double_status -eq 0 ]] || {
    echo "double-buffer pipeline failed for R$resolution" >&2
    exit "$double_status"
  }
done
