#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REPO INSTALL_ROOT PACKAGE FIXTURE SOURCE_ONNX SOURCE_COMMIT"
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# != 6 )); then usage >&2; exit 2; fi

repo=$(cd "$1" && pwd)
install_root=$(cd "$2" && pwd)
package=$(cd "$3" && pwd)
fixture=$(realpath "$4")
source_model=$(realpath "$5")
source_commit=$6
# shellcheck source=../../config/release.env
source "$repo/config/release.env"

readonly runtime_root=$Y26_RUNTIME_RELEASE_ROOT
readonly internal_root=$Y26_INTERNAL_RD_RELEASE_ROOT
readonly release_parent=$(dirname "$runtime_root")
readonly runtime_archive_base=$Y26_RUNTIME_ARCHIVE_BASE
readonly internal_archive_base=$Y26_INTERNAL_RD_ARCHIVE_BASE
opencv_prefix=${Y26_OPENCV_PREFIX:-/data/opencv/install-k1x-gtk3}
report_root=${Y26_STAGE59_REPORT_ROOT:-}
media_root=${Y26_STAGE59_MEDIA_ROOT:-}

[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit" >&2; exit 2; }
test -f "$fixture"
test -f "$source_model"
[[ $(basename "$source_model") == "$Y26_SOURCE_MODEL_FILENAME" ]] || {
  echo "unexpected source-model filename" >&2; exit 1;
}
[[ $(sha256sum "$source_model" | awk '{print $1}') == "$Y26_SOURCE_MODEL_SHA256" ]] || {
  echo "source-model SHA-256 mismatch" >&2; exit 1;
}
manifest_sha=$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')
[[ $manifest_sha == "$Y26_EXPECTED_MANIFEST_SHA256" ]] || {
  echo "package-manifest SHA-256 mismatch" >&2; exit 1;
}

install_so=$install_root/lib/liby26_k1x_int8_executor.so.$Y26_RELEASE_VERSION
for path in \
  "$install_so" \
  "$install_root/lib/liby26_k1x_int8_executor.a" \
  "$install_root/bin/yolo26_k1x_int8" \
  "$install_root/bin/y26_k1x_healthcheck" \
  "$install_root/bin/y26_k1x_demo" \
  "$install_root/bin/y26_v4l2_probe"; do
  test -f "$path"
done
capability_marker="$Y26_RELEASE_VERSION/$Y26_FULL_GRAPH_PROFILE_ID/abi$Y26_ABI_VERSION/ime1/rvv1/frozen1"
grep -aFq "$capability_marker" \
  "$install_so" || { echo "refusing non-official K1X release library" >&2; exit 1; }

rm -rf --one-file-system "$runtime_root" "$internal_root"
mkdir -p "$runtime_root"/{bin,lib/cmake,lib/pkgconfig,include,package,model,labels,fixtures,config,scripts,examples,docs,licenses,sbom,opencv/lib,outputs/{correctness,accuracy,performance,camera,screenshots,demo-video}}

for binary in yolo26_k1x_int8 y26_k1x_healthcheck y26_k1x_demo y26_v4l2_probe; do
  install -m 0755 "$install_root/bin/$binary" "$runtime_root/bin/"
done
install -m 0644 "$install_root/lib/liby26_k1x_int8_executor.a" "$runtime_root/lib/"
for name in liby26_k1x_int8_executor.so \
            liby26_k1x_int8_executor.so.$Y26_SOVERSION \
            liby26_k1x_int8_executor.so.$Y26_RELEASE_VERSION; do
  install -m 0644 -T "$install_so" "$runtime_root/lib/$name"
done
cp -aL "$install_root/lib/cmake/." "$runtime_root/lib/cmake/"
cp -aL "$install_root/lib/pkgconfig/." "$runtime_root/lib/pkgconfig/"
install -m 0644 "$install_root/include/y26_k1x_executor.h" "$runtime_root/include/"
cp -a "$package/." "$runtime_root/package/"
install -m 0644 "$repo/assets/coco80.txt" "$runtime_root/labels/"
install -m 0644 "$fixture" "$runtime_root/fixtures/bus_640_nchw_f32.bin"
install -m 0644 "$repo/config/release.env" "$runtime_root/config/"
install -m 0644 "$repo/config/k1x-int8-executor-safe.conf" "$runtime_root/config/"
cp -aL "$install_root/share/y26-k1x-int8-executor/examples/." "$runtime_root/examples/"

for module in core imgproc imgcodecs highgui videoio; do
  opencv_so=$opencv_prefix/lib/libopencv_${module}.so.4.13.0
  test -f "$opencv_so"
  install -m 0644 -T "$opencv_so" "$runtime_root/opencv/lib/libopencv_${module}.so.413"
done
install -m 0644 "$repo/INTERNAL_USE_NOTICE.md" "$runtime_root/licenses/PROJECT_INTERNAL_USE_NOTICE.md"
install -m 0644 "$repo/docs/K1X_INT8_EXECUTOR_NOTICES.md" "$runtime_root/licenses/THIRD_PARTY_NOTICES.md"
install -m 0644 /data/opencv/LICENSE "$runtime_root/licenses/OPENCV_LICENSE.txt"
install -m 0644 \
  "$repo/.deps/venvs/ultralytics_latest/lib/python3.12/site-packages/ultralytics-8.4.82.dist-info/licenses/LICENSE" \
  "$runtime_root/licenses/ULTRALYTICS_AGPL-3.0.txt"

find "$repo/docs" -maxdepth 1 -type f -name '*.md' -print0 | while IFS= read -r -d '' file; do
  install -m 0644 "$file" "$runtime_root/docs/"
done
for file in MODEL_CARD_RU.md MODEL_CARD_EN.md MODEL_SOURCE_NOT_REDISTRIBUTED.md \
            MODEL_PROVENANCE.md MODEL_LICENSE_RECORD.md SOURCE_MODEL_SHA256; do
  install -m 0644 "$repo/model/$file" "$runtime_root/model/"
done

find "$repo/scripts/k1x-int8-executor" -maxdepth 1 -type f -name '*.sh' -print0 | \
  while IFS= read -r -d '' file; do install -m 0755 "$file" "$runtime_root/scripts/"; done
for file in run_image_demo.sh run_camera_demo.sh run_camera_demo_fast.sh \
            run_camera_demo_o2_diagnostic.sh bench_forward_only.sh bench_full_demo.sh \
            detect_camera_formats.sh capture_camera_affinity.sh y26_executor_common.sh \
            verify-system-dependencies.sh camera-system-profile.sh; do
  install -m 0755 "$repo/scripts/$file" "$runtime_root/scripts/"
done

printf 'fixture\tinput\toutput_schema\texpected_output_hash\tstatus\n' \
  >"$runtime_root/outputs/correctness/known_fixture.tsv"
printf 'bus\tfixtures/bus_640_nchw_f32.bin\t1x300x6\t%s\texact\n' \
  "$Y26_EXPECTED_OUTPUT_HASH" >>"$runtime_root/outputs/correctness/known_fixture.tsv"
printf 'surface\timages\tmap50_95\tmap50\tap_small\tap_medium\tap_large\tprediction_sha256\n' \
  >"$runtime_root/outputs/accuracy/full_coco_summary.tsv"
printf 'COCO_val2017\t5000\t0.3707408944391919\t0.5258465300872381\t0.18397294626227842\t0.4142627352606523\t0.5440433811804918\t%s\n' \
  "$Y26_EXPECTED_PREDICTION_SHA256" >>"$runtime_root/outputs/accuracy/full_coco_summary.tsv"
if [[ -n $report_root && -d $report_root ]]; then
  find "$report_root" -maxdepth 1 -type f \( -name 'final_*.tsv' -o -name '*camera*.md' \) -print0 | \
    while IFS= read -r -d '' file; do install -m 0644 "$file" "$runtime_root/outputs/performance/"; done
fi
if [[ -n $media_root && -d $media_root ]]; then
  for media_kind in camera screenshots demo-video; do
    if [[ -d $media_root/$media_kind ]]; then
      cp -aL "$media_root/$media_kind/." "$runtime_root/outputs/$media_kind/"
    fi
  done
fi

install -m 0644 "$repo/docs/SUPPORTED_PLATFORM.md" "$runtime_root/SUPPORTED_PLATFORM.md"
needed_tmp=$(mktemp)
bundled_tmp=$(mktemp)
trap 'rm -f "$needed_tmp" "$bundled_tmp"' EXIT
find "$runtime_root/lib" "$runtime_root/opencv/lib" -maxdepth 2 -type f -printf '%f\n' | sort -u >"$bundled_tmp"
find "$runtime_root/bin" "$runtime_root/lib" "$runtime_root/opencv/lib" -maxdepth 2 -type f -print0 | \
  while IFS= read -r -d '' elf; do
    readelf -d "$elf" 2>/dev/null | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' || true
  done | sort -u >"$needed_tmp"
{
  printf 'soname\trole\n'
  while IFS= read -r soname; do
    grep -Fx "$soname" "$bundled_tmp" >/dev/null || printf '%s\tsupported-board-system\n' "$soname"
  done <"$needed_tmp"
} >"$runtime_root/required-system-sonames.tsv"
install -m 0755 "$repo/scripts/verify-system-dependencies.sh" \
  "$runtime_root/scripts/verify-system-dependencies.sh"

python3 "$repo/custom_int8_engine/tools/stage59_release_bundle.py" \
  --root "$runtime_root" --bundle-kind runtime --source-commit "$source_commit" \
  --release-version "$Y26_RELEASE_VERSION" \
  --model-sha256 "$Y26_SOURCE_MODEL_SHA256" \
  --integer-contract-id "$Y26_INTEGER_CONTRACT_ID" \
  --full-graph-profile-id "$Y26_FULL_GRAPH_PROFILE_ID" \
  --package-manifest-sha256 "$manifest_sha" \
  --prediction-sha256 "$Y26_EXPECTED_PREDICTION_SHA256" \
  --known-output-hash "$Y26_EXPECTED_OUTPUT_HASH"
(cd "$runtime_root" && sha256sum -c SHA256SUMS)

cp -a "$runtime_root" "$internal_root"
rm -f "$internal_root/model/MODEL_SOURCE_NOT_REDISTRIBUTED.md"
install -m 0644 "$source_model" "$internal_root/model/$Y26_SOURCE_MODEL_FILENAME"
install -m 0644 "$repo/model/INTERNAL_R&D_ONLY.md" "$internal_root/INTERNAL_R&D_ONLY.md"
install -m 0644 "$repo/model/INTERNAL_R&D_ONLY.md" "$internal_root/model/"
python3 "$repo/custom_int8_engine/tools/stage59_release_bundle.py" \
  --root "$internal_root" --bundle-kind internal-rd --source-commit "$source_commit" \
  --release-version "$Y26_RELEASE_VERSION" \
  --model-sha256 "$Y26_SOURCE_MODEL_SHA256" \
  --integer-contract-id "$Y26_INTEGER_CONTRACT_ID" \
  --full-graph-profile-id "$Y26_FULL_GRAPH_PROFILE_ID" \
  --package-manifest-sha256 "$manifest_sha" \
  --prediction-sha256 "$Y26_EXPECTED_PREDICTION_SHA256" \
  --known-output-hash "$Y26_EXPECTED_OUTPUT_HASH"
(cd "$internal_root" && sha256sum -c SHA256SUMS)

source_date_epoch=$(git -C "$repo" show -s --format=%ct "$source_commit")
make_archives() {
  local root=$1 base=$2 temp archive_root tar_path zip_path
  temp=$(mktemp -d "$release_parent/.stage59-archive.XXXXXX")
  archive_root=$temp/$base
  cp -a "$root" "$archive_root"
  find "$archive_root" -exec touch -h -d "@$source_date_epoch" {} +
  tar_path=$release_parent/$base.tar.gz
  zip_path=$release_parent/$base.zip
  rm -f "$tar_path" "$zip_path"
  tar --sort=name --mtime="@$source_date_epoch" --owner=0 --group=0 --numeric-owner \
    -C "$temp" -cf - "$base" | gzip -n >"$tar_path"
  (cd "$temp" && find "$base" -print | LC_ALL=C sort | zip -X -q "$zip_path" -@)
  rm -rf --one-file-system "$temp"
  sha256sum "$tar_path" "$zip_path"
}

make_archives "$runtime_root" "$runtime_archive_base"
make_archives "$internal_root" "$internal_archive_base"
