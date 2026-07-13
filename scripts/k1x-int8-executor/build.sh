#!/usr/bin/env bash
set -euo pipefail

repo=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
build_root=${2:-"$repo/.deps/custom_int8_engine/release-build"}
install_root=${3:-"$build_root/install"}
opencv_dir=${Y26_K1X_OPENCV_DIR:-/data/opencv/install-k1x-gtk3/lib/cmake/opencv4}
executor_flags=${Y26_K1X_EXECUTOR_CXX_FLAGS:-"-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops"}

source /data/build_scripts/01-env.sh
common=(
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
  -DY26_K1X_ENABLE_IME=ON
  -DY26_K1X_ENABLE_TESTS=OFF
  -DY26_K1X_BUILD_TOOLS=ON
  -DY26_K1X_OPENCV_DIR="$opencv_dir"
  -DCMAKE_CXX_FLAGS="$executor_flags"
)

cmake -S "$repo/custom_int8_engine" -B "$build_root/static" \
  "${common[@]}" -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX="$install_root"
cmake --build "$build_root/static" -j"${JOBS:-4}"
cmake --install "$build_root/static"

cmake -S "$repo/custom_int8_engine" -B "$build_root/shared" \
  "${common[@]}" -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$install_root"
cmake --build "$build_root/shared" -j"${JOBS:-4}"
cmake --install "$build_root/shared"

strip_tool=${STRIP:-riscv64-unknown-linux-gnu-strip}
command -v "$strip_tool" >/dev/null
"$strip_tool" -D -g "$install_root/lib/liby26_k1x_int8_executor.a"
"$strip_tool" --strip-debug \
  "$install_root/lib/liby26_k1x_int8_executor.so" \
  "$install_root/bin/yolo26_k1x_int8" \
  "$install_root/bin/y26_k1x_c_api_smoke"

find "$install_root" -type f -print0 | sort -z | xargs -0 sha256sum
