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
  /data/releases/banana-yolo26-k1x-int8-executor/0.9.0-stage57-final-handoff) ;;
  *) echo "refusing unexpected Stage57 release path: $output" >&2; exit 2 ;;
esac
test -f "$fixture"
test -f "$package/asset_hashes.tsv"
test -x "$install_root/bin/yolo26_k1x_int8"
test -x "$install_root/bin/y26_k1x_healthcheck"
package_manifest_sha256=$(sha256sum "$package/asset_hashes.tsv" | awk '{print $1}')
expected=fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be
[[ $package_manifest_sha256 == "$expected" ]] || {
  echo "unexpected package manifest: $package_manifest_sha256" >&2
  exit 1
}

rm -rf --one-file-system "$output"
mkdir -p "$output"/{bin,lib/cmake,lib/pkgconfig,include,package,fixtures,config,docs,scripts,examples,licenses,sbom,outputs}

install -m 0755 "$install_root/bin/yolo26_k1x_int8" "$output/bin/"
install -m 0755 "$install_root/bin/y26_k1x_healthcheck" "$output/bin/"
install -m 0644 "$install_root/lib/liby26_k1x_int8_executor.a" "$output/lib/"
for name in liby26_k1x_int8_executor.so liby26_k1x_int8_executor.so.1 liby26_k1x_int8_executor.so.0.9.0; do
  install -m 0644 -T "$install_root/lib/liby26_k1x_int8_executor.so.0.9.0" "$output/lib/$name"
done
cp -aL "$install_root/lib/cmake/." "$output/lib/cmake/"
cp -aL "$install_root/lib/pkgconfig/." "$output/lib/pkgconfig/"
install -m 0644 "$install_root/include/y26_k1x_executor.h" "$output/include/"
cp -a "$package/." "$output/package/"
install -m 0644 "$fixture" "$output/fixtures/bus_640_nchw_f32.bin"
install -m 0644 "$repo/config/k1x-int8-executor-safe.conf" "$output/config/"
cp -aL "$install_root/share/y26-k1x-int8-executor/examples/." "$output/examples/"

docs=(
  README_K1X_INT8_EXECUTOR.md HANDOFF_EN.md HANDOFF_RU.md QUICKSTART_RU.md
  INTEGRATION_GUIDE.md RELEASE_PROFILES.md SYSTEM_PROFILE_O2.md
  PERFORMANCE_AND_ACCURACY.md TROUBLESHOOTING_HANDOFF.md
  RELEASE_NOTES_0.9.0.md CURRENT_GRAPH_FREEZE.md
  K1X_INT8_EXECUTOR_ARCHITECTURE.md K1X_INT8_MODEL_PACKAGE_FORMAT.md
  K1X_INT8_EXECUTOR_LIMITATIONS.md K1X_INT8_EXECUTOR_NOTICES.md
  K1X_INT8_EXECUTOR_HANDOFF_CHECKLIST.md
)
for file in "${docs[@]}"; do
  install -m 0644 "$repo/docs/$file" "$output/docs/"
done

scripts=(build.sh package.sh deploy.sh smoke-test.sh benchmark.sh uninstall.sh create-release.sh o2-system-profile.sh)
for file in "${scripts[@]}"; do
  install -m 0755 "$repo/scripts/k1x-int8-executor/$file" "$output/scripts/"
done
install -m 0644 "$repo/docs/K1X_INT8_EXECUTOR_NOTICES.md" \
  "$output/licenses/THIRD_PARTY_NOTICES.md"

cat >"$output/outputs/known_fixture.tsv" <<'EOF'
fixture	input	output_schema	expected_output_hash	status
bus	fixtures/bus_640_nchw_f32.bin	1x300x6	0xd43f5e018b415631	exact
EOF

cat >"$output/outputs/accuracy_summary.tsv" <<'EOF'
surface	images	map50_95	map50	ap_small	ap_medium	ap_large	prediction_sha256
COCO_val2017	5000	0.3707408944391919	0.5258465300872381	0.18397294626227842	0.4142627352606523	0.5440433811804918	cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda
EOF

cat >"$output/outputs/performance_summary.tsv" <<'EOF'
surface	samples	mean_us	median_us	p95_us	p99_us	p999_us	max_us	notes
compatibility_fixed	500	147156.530	146493.000	151343.100	157373.510	162881.304	164031.000	condition-variable; original OS
low_latency_fixed	500	133674.926	133479.500	135944.150	136817.000	138291.851	139465.000	frame-gated spin; original OS
low_latency_dedicated_o2_fixed	500	133305.232	133307.500	133825.050	134031.490	134956.415	135413.000	frame-gated spin; reversible O2
compatibility_soak	10000	147746.663	147181.500	151137.950	157663.060	163959.195	181280.000	separate long-run surface
low_latency_dedicated_o2_soak	13500	135040.533	134995.000	135637.000	136675.070	138660.577	140242.000	separate long-run surface
real_corpus_executor	100	132913.617						100 preloaded images
rgb_input	500	131318.676	131307.000	131736.550	132003.130	133907.467	134390.000	RGB8; no JPEG/resize
serial_preloaded_pipeline	500	188654.187		191797.338	197377.444		238273.356	CPU4 preprocessing; OpenCV inherited default
double_buffer_interval	500	140555.108		144883.562			155869.738	CPU5-7 preprocessing; OpenCV 3 threads; throughput surface
matched_b120_ort	500	463234.271	463109.082	466201.426	469321.221	473800.842	473991.508	per-inference distribution
EOF

python3 "$repo/custom_int8_engine/tools/stage52_release_bundle.py" \
  --root "$output" \
  --source-commit "$source_commit" \
  --package-manifest-sha256 "$package_manifest_sha256" \
  --release-id banana-yolo26-k1x-int8-executor-0.9.0-stage57-final-handoff \
  --release-version 0.9.0 \
  --prediction-sha256 cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda \
  --known-output-hash 0xd43f5e018b415631
(cd "$output" && sha256sum -c SHA256SUMS)
