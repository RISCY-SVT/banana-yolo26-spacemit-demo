#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
root=$STAGE65D_BOARD_ROOT
stage62=/data/k1x-stage-runs/BANANA-YOLO26-K1X-STAGE62-FINAL-BRANCH-CONSOLIDATION-AGPL-INTERNAL-RD-SDK-AND-DUAL-REMOTE-FREEZE-GATE-001
custom="$stage62/release/extracted-final-9f88644/banana-yolo26-k1x-int8-executor-0.10.0-internal-rd.1-sdk-riscv64"
out="$root/custom-context"
fixture="$root/fixtures/fixed/bus_r640_nchw_f32.bin"
runner="$root/bin/stage64_two_stage_runner"
tail="$root/models/stage65b_r1_b2.postprocess.onnx"
runtime="$root/runtime/lib"
ulimit -c 0

[[ ! -e $out ]] || { printf 'custom context root exists: %s\n' "$out" >&2; exit 2; }
for path in "$custom/bin/yolo26_k1x_int8" "$custom/package/asset_hashes.tsv" \
  "$custom/release_manifest.json" "$custom/SHA256SUMS" "$fixture"; do
  [[ -r $path ]] || { printf 'missing accepted custom input: %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$out"

check_hash() {
  local label=$1 path=$2 expected=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$path" "$actual" "$expected" \
    "$([[ $actual == "$expected" ]] && echo pass || echo fail)" >>"$out/identity.raw.tsv"
  [[ $actual == "$expected" ]]
}

printf 'field\tpath\tactual\texpected\tstatus\n' >"$out/identity.raw.tsv"
check_hash release_manifest "$custom/release_manifest.json" dced8ddfc540ab5b7fd72ecfe7a16021338ea56258fb33d09c5e023ba3d98b98
check_hash sha256sums "$custom/SHA256SUMS" f9c604d7a3167664a86c48dd101e4f4935a243bde726c6853a9f9390aa278341
check_hash executor "$custom/bin/yolo26_k1x_int8" 34da155ed02a83a74babbec30aff960bdccfb6cc16018230ae7bc030462f7187
check_hash common_bus_tensor "$fixture" 64d11ef4c1e470282a385f7d293607b639da2f40405c92238897253dd1e23f14

snapshot() {
  local file=$1
  {
    printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      [[ -r $path ]] || continue
      printf 'governor\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
      [[ -r $path ]] || continue
      printf 'frequency_khz\t%s\t%s\n' "$path" "$(cat "$path")"
    done
    for path in /sys/class/thermal/thermal_zone*/temp; do
      [[ -r $path ]] || continue
      printf 'temperature_millic\t%s\t%s\n' "$path" "$(cat "$path")"
    done
  } >"$file"
}

run_vendor() {
  local model=$1 inference directory
  case "$model" in
    B2) inference="$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) inference="$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
  directory="$out/vendor-$model"
  mkdir -p "$directory/tmp" "$directory/cache"
  snapshot "$directory/state-before.tsv"
  env LD_LIBRARY_PATH="$runtime" TMPDIR="$directory/tmp" XDG_CACHE_HOME="$directory/cache" \
    taskset -c 0-3 "$runner" \
      --provider spacemit --inference-model "$inference" --tail-model "$tail" \
      --input "$fixture" --output "$directory/output.bin" \
      --samples-output "$directory/samples.tsv" --intra-threads 4 --inter-threads 1 \
      --warmup 10 --runs 100 --repeats 5 >"$directory/run.log" 2>&1
  snapshot "$directory/state-after.tsv"
  sha256sum "$directory/output.bin" "$directory/samples.tsv" >"$directory/output-sha256.txt"
}

run_custom() {
  local directory="$out/custom"
  mkdir -p "$directory"
  snapshot "$directory/state-before.tsv"
  env LD_LIBRARY_PATH="$custom/lib:$custom/opencv/lib" \
    taskset -c 0-4 "$custom/bin/yolo26_k1x_int8" \
      --package "$custom/package" --image "$fixture" --input-mode preprocessed-f32 \
      --profile low-latency --warmup 10 --runs 100 --repeats 5 \
      --benchmark --verify-determinism \
      --expected-manifest-sha256 fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
      >"$directory/samples.raw.tsv" 2>"$directory/run.stderr"
  snapshot "$directory/state-after.tsv"
  sha256sum "$directory/samples.raw.tsv" >"$directory/output-sha256.txt"
}

run_vendor B2
run_vendor C2
run_custom
printf 'stage65d_r1_custom_context status=pass\n'
