#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../config/release.env
source "$repo/config/release.env"
build_root=${Y26_BUILD_ROOT:-/data/build/banana-yolo26-k1x-demo-$Y26_RELEASE_VERSION}
install_root=${Y26_INSTALL_ROOT:-/data/install/banana-yolo26-k1x-demo-$Y26_RELEASE_VERSION}

usage() { echo "usage: $0 [--build-root DIR] [--install-root DIR]"; }
while (($#)); do
  case $1 in
    --build-root) build_root=$2; shift 2 ;;
    --install-root) install_root=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

source /data/build_scripts/01-env.sh
"$repo/scripts/ensure_opencv.sh"
source_commit=$(git -C "$repo" rev-parse HEAD)
cmake -S "$repo" -B "$build_root" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$repo/cmake/toolchains/k1x-spacemit-cross.cmake" \
  -DCMAKE_INSTALL_PREFIX="$install_root" \
  -DOpenCV_DIR=/data/opencv/install-k1x-gtk3/lib/cmake/opencv4 \
  -DY26_K1X_ENABLE_IME=ON \
  -DY26_DEMO_OFFICIAL_K1X_RELEASE=ON \
  -DY26_K1X_SOURCE_COMMIT="$source_commit"
cmake --build "$build_root" -j"${JOBS:-$(nproc)}"
cmake --install "$build_root"

so=$install_root/lib/liby26_k1x_int8_executor.so.$Y26_RELEASE_VERSION
capability_marker="$Y26_RELEASE_VERSION/$Y26_FULL_GRAPH_PROFILE_ID/abi$Y26_ABI_VERSION/ime1/rvv1/frozen1"
grep -aFq "$capability_marker" "$so" || {
  echo "official release build lacks IME/RVV/frozen-profile capabilities" >&2
  exit 1
}
file "$install_root/bin/y26_k1x_demo" "$so"
find "$install_root" -type f -print0 | sort -z | xargs -0 sha256sum
