#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
root=$STAGE65D_BOARD_ROOT
runtime="$root/runtime/lib"
runner="$root/bin/stage64_two_stage_runner"
wrapper="$root/bin/stage65d_run_two_stage_board.sh"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
ulimit -c 0
cd "$root"

mkdir -p "$root/profile" "$root/fixed-fixtures"
printf 'surface\tmodel\tprovider\tmode\tfixture\texit_code\toutput_sha256\tlog\n' \
  >"$root/profile/session_matrix.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

run_surface() {
  local model=$1
  local provider=$2
  local mode=$3
  local fixture=$4
  local input=$5
  local output_dir=$6
  local inference
  inference=$(inference_for "$model")
  local rc=0
  local profile_flag=()
  local warmup=0 runs=1 repeats=1
  if [[ $mode == profile ]]; then
    profile_flag=(--enable-profiling)
  elif [[ $mode == repeat ]]; then
    warmup=2
    runs=3
  fi
  mkdir -p "$output_dir/provider-dumps"
  set +e
  env \
    SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
    SPACEMIT_EP_DUMP_SUBGRAPHS_DIR="$output_dir/provider-dumps" \
    timeout --signal=TERM --kill-after=10s 600s \
    "$wrapper" \
      --runner "$runner" \
      --runtime-lib "$runtime" \
      --inference "$inference" \
      --tail "$tail_model" \
      --input "$input" \
      --output-dir "$output_dir" \
      --provider "$provider" \
      --cpu-list 0-3 \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup "$warmup" \
      --runs "$runs" \
      --repeats "$repeats" \
      "${profile_flag[@]}"
  rc=$?
  set -e
  local output_hash=missing
  [[ -f $output_dir/output.bin ]] && output_hash=$(sha256sum "$output_dir/output.bin" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}_${provider}_${mode}_${fixture}" "$model" "$provider" "$mode" \
    "$fixture" "$rc" "$output_hash" "$output_dir/run.log" \
    >>"$root/profile/session_matrix.raw.tsv"
  return "$rc"
}

for model in B2 C2; do
  for provider in cpu spacemit; do
    for mode in profile repeat; do
      run_surface "$model" "$provider" "$mode" F0 \
        "$root/fixtures/fixed/images_F0_f32.bin" \
        "$root/profile/${model}-${provider}-${mode}"
    done
  done
done

printf 'surface\tmodel\tprovider\tfixture\texit_code\toutput_sha256\tboundary_manifest_sha256\n' \
  >"$root/fixed-fixtures/fixed_fixture_results.raw.tsv"
for fixture_spec in \
  images_F0_f32:F0 \
  bus_r640_nchw_f32:bus \
  zidane_r640_nchw_f32:zidane \
  canonical_640_nchw_f32:canonical; do
  fixture=${fixture_spec%%:*}
  label=${fixture_spec#*:}
  input="$root/fixtures/fixed/${fixture}.bin"
  for model in B2 C2; do
    for provider in cpu spacemit; do
      output_dir="$root/fixed-fixtures/${model}-${provider}-${label}"
      rc=0
      run_surface "$model" "$provider" fixed "$label" "$input" "$output_dir" || rc=$?
      boundary_manifest=missing
      if compgen -G "$output_dir/boundaries/*.bin" >/dev/null; then
        boundary_manifest=$(
          sha256sum "$output_dir"/boundaries/*.bin | sha256sum | awk '{print $1}'
        )
      fi
      output_hash=missing
      [[ -f $output_dir/output.bin ]] && output_hash=$(sha256sum "$output_dir/output.bin" | awk '{print $1}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model}_${provider}_${label}" "$model" "$provider" "$label" "$rc" \
        "$output_hash" "$boundary_manifest" \
        >>"$root/fixed-fixtures/fixed_fixture_results.raw.tsv"
      [[ $rc -eq 0 ]] || exit "$rc"
    done
  done
done

printf 'stage65d_board_sessions status=pass\n'
