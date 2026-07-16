#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [REPO [BUILD_ROOT [INSTALL_ROOT]]]"
  exit 0
fi
repo=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
build_root=${2:-"$repo/.deps/custom_int8_engine/release-build"}
install_root=${3:-"$build_root/install"}
executor_flags=${Y26_K1X_EXECUTOR_CXX_FLAGS:-"-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops"}

source /data/build_scripts/01-env.sh
source_commit=$(git -C "$repo" rev-parse HEAD)
cmake -S "$repo/custom_int8_engine" -B "$build_root" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_PREFIX="$install_root" \
  -DCMAKE_CXX_FLAGS="$executor_flags" \
  -DY26_K1X_ENABLE_IME=ON \
  -DY26_K1X_ENABLE_TESTS=OFF \
  -DY26_K1X_BUILD_RELEASE=ON \
  -DY26_K1X_BUILD_RELEASE_TOOLS=ON \
  -DY26_K1X_BUILD_RESEARCH=OFF \
  -DY26_K1X_BUILD_TOOLS=OFF \
  -DY26_K1X_INSTALL_DEVELOPMENT_HEADERS=OFF \
  -DY26_K1X_SOURCE_COMMIT="$source_commit"
cmake --build "$build_root" -j"${JOBS:-4}"
cmake --install "$build_root"

strip_tool=${STRIP:-riscv64-unknown-linux-gnu-strip}
command -v "$strip_tool" >/dev/null
"$strip_tool" -D -g "$install_root/lib/liby26_k1x_int8_executor.a"
"$strip_tool" --strip-debug \
  "$install_root/lib/liby26_k1x_int8_executor.so.0.9.1" \
  "$install_root/bin/yolo26_k1x_int8" \
  "$install_root/bin/y26_k1x_healthcheck"

strings "$install_root/lib/liby26_k1x_int8_executor.so.0.9.1" | \
  grep -Fq '0.9.1/K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001/abi1/ime1/rvv1/frozen1' || {
    echo "official K1X release capability marker is missing" >&2
    exit 1
  }

find "$install_root" -type f -print0 | sort -z | xargs -0 sha256sum
