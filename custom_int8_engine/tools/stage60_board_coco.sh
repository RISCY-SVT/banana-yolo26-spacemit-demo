#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE_ROOT RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
shift
ram=/dev/shm/y26-stage60-coco-$$
mkdir -p "$ram" "$stage_root/predictions"
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

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  prediction=$ram/r${resolution}_predictions.json
  timing=$ram/r${resolution}_timing.tsv
  summary=$ram/r${resolution}_summary.txt
  "$stage_root/bin/stage52_coco_predict" \
    --package "$package" \
    --images "$stage_root/coco/val2017" \
    --output "$prediction" \
    --timing-tsv "$timing" \
    --threads 4 --limit 0 --log-every 250 --conf 0.001 \
    >"$summary" 2>&1
  sha256sum "$prediction" "$timing" >>"$summary"
  cp "$prediction" "$stage_root/predictions/r${resolution}_predictions.json"
  cp "$timing" "$stage_root/predictions/r${resolution}_timing.tsv"
  cp "$summary" "$stage_root/predictions/r${resolution}_summary.txt"
done
