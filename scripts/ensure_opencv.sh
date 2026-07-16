#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0  # verify the canonical K1X OpenCV 4.13 prefix"
  exit 0
fi
[[ $# == 0 ]] || { echo "usage: $0" >&2; exit 2; }
prefix=${Y26_OPENCV_PREFIX:-/data/opencv/install-k1x-gtk3}
test -f "$prefix/lib/cmake/opencv4/OpenCVConfig.cmake"
for module in core imgproc imgcodecs highgui videoio; do
  test -f "$prefix/lib/libopencv_${module}.so.4.13.0" || {
    echo "missing K1X OpenCV module: $module" >&2
    exit 1
  }
done
echo "OpenCV K1X prefix verified: $prefix"
