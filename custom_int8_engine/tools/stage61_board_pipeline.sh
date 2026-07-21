#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 STAGE_ROOT IMAGE RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
image=$2
shift 2
ram=/dev/shm/y26-stage61-pipeline-$$
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
export Y26_STAGE61_ATTENTION_NTAIL=1

opencv_lib=${Y26_STAGE61_OPENCV_LIB:-$stage_root/lib/opencv}
[[ -d $opencv_lib ]] || {
  echo "Stage61 OpenCV runtime closure is missing: $opencv_lib" >&2
  exit 1
}
export LD_LIBRARY_PATH=$opencv_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  [[ -r $package/asset_hashes.tsv ]] || {
    echo "Stage61 package is missing for R$resolution" >&2
    exit 1
  }
  for mode in serial double-buffer; do
    prefix=$ram/r${resolution}_${mode}
    file_mode=${mode//-/_}
    if [[ $mode == serial ]]; then
      command=("$stage_root/bin/stage52_image_pipeline_bench"
        --package "$package" --image "$image" --threads 4
        --warmup 10 --runs 100 --repeats 5)
    else
      command=("$stage_root/bin/stage56_double_buffer_bench"
        --package "$package" --image "$image"
        --warmup 10 --runs 100 --repeats 5)
    fi
    set +e
    "${command[@]}" >"$prefix.tsv" 2>"$prefix.stderr"
    status=$?
    set -e
    printf '%s\n' "$status" >"$prefix.exit-status.txt"
    cp "$prefix.tsv" "$output/r${resolution}_${file_mode}.tsv"
    cp "$prefix.stderr" "$output/r${resolution}_${file_mode}.stderr"
    cp "$prefix.exit-status.txt" "$output/r${resolution}_${file_mode}.exit-status.txt"
    [[ $status -eq 0 ]] || {
      echo "Stage61 $mode pipeline failed for R$resolution" >&2
      exit "$status"
    }
  done
done
