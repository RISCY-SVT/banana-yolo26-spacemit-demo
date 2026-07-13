#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
  echo "usage: $0 REPO INSTALL_ROOT PACKAGE FIXTURE OUTPUT SOURCE_COMMIT" >&2
  exit 2
fi
repo=$(cd "$1" && pwd)
install_root=$(cd "$2" && pwd)
package=$(cd "$3" && pwd)
fixture=$4
output=$5
source_commit=$6

case "$output" in
  /data/releases/banana-yolo26-k1x-int8-executor|/data/releases/banana-yolo26-k1x-int8-executor/*) ;;
  *) echo "refusing release path outside /data/releases/banana-yolo26-k1x-int8-executor" >&2; exit 2 ;;
esac
test -f "$fixture"
test -f "$package/asset_hashes.tsv"
package_manifest_sha256=$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')

rm -rf --one-file-system "$output"
mkdir -p "$output"/{bin,lib,include,package,fixtures,docs,scripts,outputs,opencv/lib,licenses}
cp -a "$install_root/bin/." "$output/bin/"
cp -a "$install_root/lib/." "$output/lib/"
cp -a "$install_root/include/." "$output/include/"
cp -a "$package/." "$output/package/"
cp -a "$fixture" "$output/fixtures/bus_640_nchw_f32.bin"
cp -a "$repo"/docs/{README_K1X_INT8_EXECUTOR.md,BUILDING_K1X_INT8_EXECUTOR.md,DEPLOYING_K1X_INT8_EXECUTOR.md,K1X_INT8_EXECUTOR_API.md,K1X_INT8_MODEL_PACKAGE_FORMAT.md,K1X_INT8_EXECUTOR_ARCHITECTURE.md,K1X_INT8_EXECUTOR_CORRECTNESS.md,K1X_INT8_EXECUTOR_ACCURACY.md,K1X_INT8_EXECUTOR_PERFORMANCE.md,K1X_INT8_EXECUTOR_LIMITATIONS.md,K1X_INT8_EXECUTOR_TROUBLESHOOTING.md,K1X_INT8_EXECUTOR_HANDOFF_CHECKLIST.md,K1X_INT8_EXECUTOR_NOTICES.md} "$output/docs/"
cp -a "$repo"/scripts/k1x-int8-executor/{build.sh,package.sh,deploy.sh,smoke-test.sh,benchmark.sh,uninstall.sh,create-release.sh} "$output/scripts/"
opencv_root=${Y26_K1X_OPENCV_ROOT:-/data/opencv/install-k1x-gtk3}
for library in core imgproc imgcodecs; do
  source=$(readlink -f "$opencv_root/lib/libopencv_${library}.so.413")
  test -f "$source"
  cp -a "$source" "$output/opencv/lib/libopencv_${library}.so.413"
done
cp -a /data/opencv/LICENSE "$output/licenses/OPENCV-LICENSE.txt"
chmod 0755 "$output/bin/yolo26_k1x_int8" "$output/scripts/"*.sh

python3 "$repo/custom_int8_engine/tools/stage52_release_bundle.py" \
  --root "$output" \
  --source-commit "$source_commit" \
  --package-manifest-sha256 "$package_manifest_sha256"
(cd "$output" && sha256sum -c release_sha256.txt)
