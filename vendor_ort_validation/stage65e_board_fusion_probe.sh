#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65E_BOARD_ROOT:?STAGE65E_BOARD_ROOT is required}"
root=$STAGE65E_BOARD_ROOT
runtime="$root/runtime/lib"
runner="$root/bin/vendor_single_model_runner"
perf="$root/runtime/bin/onnxruntime_perf_test"
input="$root/fixtures/fixed/images_F0_f32.bin"
output_name=/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0
out="$root/fusion-probes"
ulimit -c 0

[[ ! -e $out ]] || { printf 'fusion probe output exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$out"
printf 'model\topt_level\texit_code\toutput_sha256\tprofile_count\tprovider_dump_count\tlog_sha256\n' >"$out/optimization-status.raw.tsv"
printf 'probe\tmodel\texit_code\tartifact\tartifact_bytes\tartifact_sha256\tlog_sha256\n' >"$out/capability-status.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

hash_or_missing() {
  local path=$1
  if [[ -f $path ]]; then
    sha256sum "$path" | awk '{print $1}'
  else
    printf 'missing'
  fi
}

run_level() {
  local model=$1 level=$2 inference directory rc=0 profiles dumps
  inference=$(inference_for "$model")
  directory="$out/optimization-${model}-${level}"
  mkdir -p "$directory/tmp" "$directory/cache" "$directory/profile" "$directory/provider-dumps"
  set +e
  env LD_LIBRARY_PATH="$runtime" TMPDIR="$directory/tmp" XDG_CACHE_HOME="$directory/cache" \
    SPACEMIT_EP_DUMP_SUBGRAPHS=1 SPACEMIT_EP_DUMP_SUBGRAPHS_DIR="$directory/provider-dumps" \
    timeout --signal=TERM --kill-after=10s 900s taskset -c 0-3 "$runner" \
      --provider spacemit --model "$inference" --input "$input" --output "$directory/output.bin" \
      --input-name images --output-name "$output_name" --opt-level "$level" \
      --execution-mode sequential --intra-threads 4 --inter-threads 1 \
      --memory-pattern 1 --cpu-arena 1 --thread-spinning 0 \
      --log-severity 2 --log-verbosity 0 --warmup 5 --runs 20 --repeats 1 \
      --profile-prefix "$directory/profile/ort-profile" \
      >"$directory/run.log" 2>&1
  rc=$?
  set -e
  profiles=$(find "$directory/profile" -type f | wc -l)
  dumps=$(find "$directory/provider-dumps" -type f -name 'SpaceMITExecutionProvider_*.onnx' | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$level" "$rc" \
    "$(hash_or_missing "$directory/output.bin")" "$profiles" "$dumps" \
    "$(hash_or_missing "$directory/run.log")" >>"$out/optimization-status.raw.tsv"
}

run_capability() {
  local probe=$1 model=$2 artifact=$3
  shift 3
  local directory="$out/${probe}-${model}" rc=0 bytes=0 artifact_hash=missing
  mkdir -p "$directory/tmp" "$directory/cache"
  set +e
  (
    cd "$directory"
    env LD_LIBRARY_PATH="$runtime" TMPDIR="$directory/tmp" XDG_CACHE_HOME="$directory/cache" \
      timeout --signal=TERM --kill-after=10s 900s taskset -c 0-3 "$@"
  ) >"$directory/run.log" 2>&1
  rc=$?
  set -e
  if [[ -f $directory/$artifact ]]; then
    bytes=$(stat -c %s "$directory/$artifact")
    artifact_hash=$(hash_or_missing "$directory/$artifact")
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$probe" "$model" "$rc" \
    "$directory/$artifact" "$bytes" "$artifact_hash" "$(hash_or_missing "$directory/run.log")" \
    >>"$out/capability-status.raw.tsv"
}

for model in B2 C2; do
  for level in disable basic extended all; do
    run_level "$model" "$level"
  done
done

for model in B2 C2; do
  inference=$(inference_for "$model")
  run_capability offline-optimized "$model" optimized.onnx \
    "$perf" -e spacemit -o 99 -u optimized.onnx -m times -r 1 -x 4 -y 1 -S 65016 "$inference" perf-results.txt
  run_capability iobinding-baseline "$model" no-artifact \
    "$perf" -e spacemit -o 0 -m times -r 10 -x 4 -y 1 -S 65016 "$inference" perf-results.txt
  run_capability iobinding-input "$model" no-artifact \
    "$perf" -I -e spacemit -o 0 -m times -r 10 -x 4 -y 1 -S 65016 "$inference" perf-results.txt
  run_capability ep-context "$model" compiled-context.onnx \
    "$perf" --compile_ep_context --compile_only --compile_model_path compiled-context.onnx \
      -e spacemit -o 99 -x 4 -y 1 "$inference"
done

set +e
env LD_LIBRARY_PATH="$runtime" "$perf" --list_ep_devices >"$out/ep-devices.log" 2>&1
devices_rc=$?
set -e
printf 'ep-device-list\tall\t%s\t%s\t0\tmissing\t%s\n' "$devices_rc" "$out/ep-devices.log" \
  "$(hash_or_missing "$out/ep-devices.log")" >>"$out/capability-status.raw.tsv"

printf 'stage65e_board_fusion_probe status=complete\n'
