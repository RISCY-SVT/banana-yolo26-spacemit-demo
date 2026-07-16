#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REPO INSTALL_ROOT PACKAGE FIXTURE OUTPUT SOURCE_COMMIT"
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# != 6 )); then usage >&2; exit 2; fi

repo=$(cd "$1" && pwd)
install_root=$(cd "$2" && pwd)
package=$(cd "$3" && pwd)
fixture=$(realpath "$4")
output=$5
source_commit=$6
readonly expected_manifest=fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be
readonly prediction_sha=cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda
readonly release_parent=/data/releases/banana-yolo26-k1x-int8-executor
readonly expected_output=$release_parent/0.9.1-stage58-camera-handoff
readonly archive_basename=banana-yolo26-k1x-int8-executor-0.9.1-riscv64
opencv_prefix=${Y26_OPENCV_PREFIX:-/data/opencv/install-k1x-gtk3}
report_root=${Y26_STAGE58_REPORT_ROOT:-}
media_root=${Y26_STAGE58_MEDIA_ROOT:-/data/Screenshots/yolo26-stage58}

[[ $output == "$expected_output" ]] || {
  echo "refusing unexpected Stage58 release path: $output" >&2
  exit 2
}
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit" >&2; exit 2; }
test -f "$fixture"
test -f "$package/asset_hashes.tsv"
test -x "$install_root/bin/yolo26_k1x_int8"
test -x "$install_root/bin/y26_k1x_healthcheck"
test -x "$install_root/bin/y26_k1x_demo"
manifest_sha=$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')
[[ $manifest_sha == "$expected_manifest" ]] || {
  echo "unexpected package manifest: $manifest_sha" >&2
  exit 1
}

rm -rf --one-file-system "$output"
mkdir -p "$output"/{bin,lib/cmake,lib/pkgconfig,include,package,model,labels,fixtures,config,scripts,examples,docs,licenses,sbom,opencv/lib,outputs/{correctness,accuracy,performance,camera,screenshots,demo-video}}

install -m 0755 "$install_root/bin/yolo26_k1x_int8" "$output/bin/"
install -m 0755 "$install_root/bin/y26_k1x_healthcheck" "$output/bin/"
install -m 0755 "$install_root/bin/y26_k1x_demo" "$output/bin/"
install -m 0644 "$install_root/lib/liby26_k1x_int8_executor.a" "$output/lib/"
for name in liby26_k1x_int8_executor.so liby26_k1x_int8_executor.so.1 liby26_k1x_int8_executor.so.0.9.1; do
  install -m 0644 -T "$install_root/lib/liby26_k1x_int8_executor.so.0.9.1" "$output/lib/$name"
done
cp -aL "$install_root/lib/cmake/." "$output/lib/cmake/"
cp -aL "$install_root/lib/pkgconfig/." "$output/lib/pkgconfig/"
install -m 0644 "$install_root/include/y26_k1x_executor.h" "$output/include/"
cp -a "$package/." "$output/package/"
install -m 0644 "$repo/assets/coco80.txt" "$output/labels/"
install -m 0644 "$fixture" "$output/fixtures/bus_640_nchw_f32.bin"
install -m 0644 "$repo/config/k1x-int8-executor-safe.conf" "$output/config/"
cp -aL "$install_root/share/y26-k1x-int8-executor/examples/." "$output/examples/"

for module in core imgproc imgcodecs highgui videoio; do
  source="$opencv_prefix/lib/libopencv_${module}.so.4.13.0"
  test -f "$source"
  install -m 0644 -T "$source" "$output/opencv/lib/libopencv_${module}.so.413"
done
install -m 0644 "$repo/docs/K1X_INT8_EXECUTOR_NOTICES.md" "$output/licenses/THIRD_PARTY_NOTICES.md"
install -m 0644 /data/opencv/LICENSE "$output/licenses/OPENCV_LICENSE.txt"

docs=(
  README_K1X_INT8_EXECUTOR.md HANDOFF_EN.md HANDOFF_RU.md QUICKSTART_RU.md
  INTEGRATION_GUIDE.md RELEASE_PROFILES.md SYSTEM_PROFILE_O2.md
  PERFORMANCE_AND_ACCURACY.md TROUBLESHOOTING_HANDOFF.md CURRENT_GRAPH_FREEZE.md
  K1X_INT8_EXECUTOR_ARCHITECTURE.md K1X_INT8_MODEL_PACKAGE_FORMAT.md
  K1X_INT8_EXECUTOR_LIMITATIONS.md K1X_INT8_EXECUTOR_NOTICES.md
  K1X_INT8_EXECUTOR_HANDOFF_CHECKLIST.md BUILDING_K1X_INT8_EXECUTOR.md
  COLLEAGUE_FAQ_RU.md COLLEAGUE_FAQ_EN.md CAMERA_DEMO_RU.md CAMERA_DEMO_EN.md
  MODEL_RESOLUTION_AND_OBJECT_SIZE_RU.md MODEL_RESOLUTION_AND_OBJECT_SIZE_EN.md
  DISTRIBUTION_ARCHIVE_RU.md DISTRIBUTION_ARCHIVE_EN.md RELEASE_NOTES_0.9.1.md
)
for file in "${docs[@]}"; do install -m 0644 "$repo/docs/$file" "$output/docs/"; done
for file in MODEL_CARD_RU.md MODEL_CARD_EN.md MODEL_SOURCE_NOT_REDISTRIBUTED.md; do
  install -m 0644 "$repo/model/$file" "$output/model/"
done

executor_scripts=(build.sh package.sh deploy.sh smoke-test.sh benchmark.sh uninstall.sh create-release.sh o2-system-profile.sh)
for file in "${executor_scripts[@]}"; do
  install -m 0755 "$repo/scripts/k1x-int8-executor/$file" "$output/scripts/"
done
demo_scripts=(run_image_demo.sh run_camera_demo.sh run_camera_demo_fast.sh bench_forward_only.sh bench_full_demo.sh detect_camera_formats.sh capture_camera_affinity.sh y26_executor_common.sh)
for file in "${demo_scripts[@]}"; do install -m 0755 "$repo/scripts/$file" "$output/scripts/"; done

cat >"$output/outputs/correctness/known_fixture.tsv" <<'EOF'
fixture	input	output_schema	expected_output_hash	status
bus	fixtures/bus_640_nchw_f32.bin	1x300x6	0xd43f5e018b415631	exact
EOF
cat >"$output/outputs/accuracy/full_coco_summary.tsv" <<'EOF'
surface	images	map50_95	map50	ap_small	ap_medium	ap_large	prediction_sha256
COCO_val2017	5000	0.3707408944391919	0.5258465300872381	0.18397294626227842	0.4142627352606523	0.5440433811804918	cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda
EOF

if [[ -n $report_root && -d $report_root ]]; then
  for file in final_correctness_matrix.tsv final_coco_report.md final_coco_prediction_hashes.tsv; do
    [[ -f $report_root/$file ]] && install -m 0644 "$report_root/$file" "$output/outputs/correctness/"
  done
  for file in final_executor_performance.tsv final_camera_performance.tsv final_pipeline_performance.tsv camera_full_fps_report_en.md camera_full_fps_report_ru.md; do
    [[ -f $report_root/$file ]] && install -m 0644 "$report_root/$file" "$output/outputs/performance/"
  done
fi
if [[ -d $media_root ]]; then
  find "$media_root" -maxdepth 1 -type f -name '*.png' -print0 | sort -z | head -z -n 6 | xargs -0 -r -I{} install -m 0644 {} "$output/outputs/screenshots/"
  find "$media_root" -maxdepth 1 -type f \( -name '*.avi' -o -name '*.mp4' \) -print0 | sort -z | head -z -n 2 | xargs -0 -r -I{} install -m 0644 {} "$output/outputs/demo-video/"
fi

python3 "$repo/custom_int8_engine/tools/stage58_release_bundle.py" \
  --root "$output" --source-commit "$source_commit" \
  --package-manifest-sha256 "$manifest_sha" --prediction-sha256 "$prediction_sha" \
  --known-output-hash 0xd43f5e018b415631
(cd "$output" && sha256sum -c SHA256SUMS)

source_date_epoch=$(git -C "$repo" show -s --format=%ct "$source_commit")
archive_tmp=$(mktemp -d "$release_parent/.stage58-archive.XXXXXX")
trap 'rm -rf --one-file-system "$archive_tmp"' EXIT
archive_root="$archive_tmp/$archive_basename"
cp -a "$output" "$archive_root"
find "$archive_root" -exec touch -h -d "@$source_date_epoch" {} +
tar_path="$release_parent/$archive_basename.tar.gz"
zip_path="$release_parent/$archive_basename.zip"
tar --sort=name --mtime="@$source_date_epoch" --owner=0 --group=0 --numeric-owner \
  -C "$archive_tmp" -cf - "$archive_basename" | gzip -n >"$tar_path"
(cd "$archive_tmp" && find "$archive_basename" -print | LC_ALL=C sort | zip -X -q "$zip_path" -@)
sha256sum "$tar_path" "$zip_path"
