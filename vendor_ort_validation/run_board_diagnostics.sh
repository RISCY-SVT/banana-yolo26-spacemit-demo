#!/usr/bin/env bash
set -euo pipefail

: "${STAGE63_BOARD_ROOT:?set STAGE63_BOARD_ROOT}"

root=$STAGE63_BOARD_ROOT
input_tiny="$root/repros/input_1x3x8x8_f32.bin"
input_full="$root/repros/canonical_640_nchw_f32.bin"
mkdir -p \
  "$root/loader-negative/raw" "$root/loader-negative/outputs" \
  "$root/filtered-diagnostics/raw" "$root/filtered-diagnostics/outputs" \
  "$root/fixed-fixtures-canonical/raw" "$root/fixed-fixtures-canonical/outputs"

run_child() {
  local log=$1
  shift
  set +e
  timeout --signal=TERM --kill-after=5s 180s "$@" >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

run_loader_mismatch() {
  local runner_version=$1
  local runtime_version=$2
  local label="runner_${runner_version}_runtime_${runtime_version}"
  local output="$root/loader-negative/outputs/${label}.bin"
  local log="$root/loader-negative/raw/${label}.log"
  local runner="$root/runners/runner_${runner_version}"
  local lib="$root/runtimes/${runtime_version}/lib"
  rm -f "$output"
  local rc
  rc=$(run_child "$log" taskset -c 0 env "LD_LIBRARY_PATH=$lib" "$runner" \
    --provider spacemit --model "$root/repros/03_conv_qdq.onnx" \
    --input "$input_tiny" --output "$output" --opt-level disable \
    --intra-threads 1 --inter-threads 1 --thread-spinning 0 --warmup 0 --runs 1 --repeats 1)
  local output_sha=missing
  [[ -f "$output" ]] && output_sha=$(sha256sum "$output" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$runner_version" "$runtime_version" "$rc" "$output_sha" "$log"
}

{
  printf 'runner_header_version\truntime_library_version\texit_code\toutput_sha256\tlog\n'
  run_loader_mismatch rt204 rt205
  run_loader_mismatch rt205 rt206
  run_loader_mismatch rt206 rt205
} >"$root/loader-negative/loader_negative_controls.raw.tsv"

set +e
taskset -c 0 env LD_LIBRARY_PATH="$root/runtimes/rt206/lib" LD_DEBUG=libs \
  timeout --signal=TERM --kill-after=5s 180s \
  "$root/runners/runner_rt204" \
    --provider spacemit --model "$root/repros/03_conv_qdq.onnx" \
    --input "$input_tiny" --output "$root/loader-negative/outputs/ld_debug_mismatch.bin" \
    --opt-level disable --intra-threads 1 --inter-threads 1 --thread-spinning 0 \
    --warmup 0 --runs 1 --repeats 1 \
    >"$root/loader-negative/raw/ld_debug_mismatch.stdout" \
    2>"$root/loader-negative/raw/ld_debug_mismatch.stderr"
printf '%s\n' "$?" >"$root/loader-negative/raw/ld_debug_mismatch.exit"
set -e

run_filtered() {
  local label=$1
  local filter=$2
  local output="$root/filtered-diagnostics/outputs/${label}.bin"
  local log="$root/filtered-diagnostics/raw/${label}.log"
  local rc
  rm -f "$output"
  rc=$(
    cd "$root/filtered-diagnostics"
    run_child "$log" taskset -c 0-3 env LD_LIBRARY_PATH="$root/runtimes/rt206/lib" \
      "$root/runners/runner_rt206" --provider spacemit \
      --model "$root/models/manual_e2e_rep_conv_matmul_qdq.onnx" \
      --input "$input_full" --output "$output" --opt-level all \
      --execution-mode sequential --intra-threads 4 --inter-threads 1 \
      --thread-spinning 0 --warmup 0 --runs 1 --repeats 1 \
      --provider-option SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
      --provider-option "SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=$filter"
  )
  local output_sha=missing
  [[ -f "$output" ]] && output_sha=$(sha256sum "$output" | awk '{print $1}')
  local result=fail
  [[ "$rc" == 0 && -f "$output" ]] && result=pass
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$filter" "$rc" "$output_sha" "$result"
}

{
  printf 'arm\tdisabled_op_types\texit_code\toutput_sha256\tresult\n'
  run_filtered exclude_qdq 'QuantizeLinear;DequantizeLinear'
  run_filtered exclude_conv 'Conv'
  run_filtered exclude_matmul 'MatMul'
  run_filtered exclude_add 'Add'
  run_filtered exclude_qdq_conv_matmul_add 'QuantizeLinear;DequantizeLinear;Conv;MatMul;Add'
} >"$root/filtered-diagnostics/filtered_diagnostics.raw.tsv"

run_canonical() {
  local surface=$1
  local provider=$2
  local model
  case "$surface" in
    fp32) model="$root/models/yolo26n_640_e2e_fp32.onnx" ;;
    fp16) model="$root/models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx" ;;
    int8) model="$root/models/manual_e2e_rep_conv_matmul_qdq.onnx" ;;
    *) return 2 ;;
  esac
  local label="${surface}_${provider}"
  local output="$root/fixed-fixtures-canonical/outputs/${label}.bin"
  local log="$root/fixed-fixtures-canonical/raw/${label}.log"
  local rc
  rm -f "$output"
  rc=$(run_child "$log" taskset -c 0-3 env LD_LIBRARY_PATH="$root/runtimes/rt206/lib" \
    "$root/runners/runner_rt206" --provider "$provider" --model "$model" \
    --input "$input_full" --output "$output" --opt-level all \
    --execution-mode sequential --intra-threads 4 --inter-threads 1 \
    --thread-spinning 0 --warmup 0 --runs 1 --repeats 1)
  local output_sha=missing
  [[ -f "$output" ]] && output_sha=$(sha256sum "$output" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\n' "$surface" "$provider" "$rc" "$output_sha"
}

{
  printf 'surface\tprovider\texit_code\toutput_sha256\n'
  run_canonical fp32 cpu
  run_canonical fp32 spacemit
  run_canonical fp16 cpu
  run_canonical fp16 spacemit
  run_canonical int8 cpu
} >"$root/fixed-fixtures-canonical/canonical_fixture_results.raw.tsv"
