#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
root=$STAGE65D_BOARD_ROOT
runtime="$root/runtime/lib"
runner="$root/bin/stage64_two_stage_runner"
wrapper="$root/bin/stage65d_run_two_stage_board.sh"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
input="$root/fixtures/fixed/images_F0_f32.bin"
ulimit -c 0

out="$root/r1-recheck"
[[ ! -e $out ]] || { printf 'recheck root exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$out"
printf 'surface\tmodel\tprovider\tmode\texit_code\toutput_sha256\tboundary_manifest_sha256\tprofile_count\n' \
  >"$out/status.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

run_arm() {
  local model=$1 provider=$2 mode=$3
  local directory="$out/${model}-${provider}-${mode}"
  local inference
  inference=$(inference_for "$model")
  local profile_args=()
  if [[ $mode == profile ]]; then
    profile_args=(--enable-profiling)
  fi
  mkdir -p "$directory/provider-dumps"
  local rc=0
  set +e
  env \
    SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
    SPACEMIT_EP_DUMP_SUBGRAPHS_DIR="$directory/provider-dumps" \
    timeout --signal=TERM --kill-after=10s 600s \
    "$wrapper" \
      --runner "$runner" \
      --runtime-lib "$runtime" \
      --inference "$inference" \
      --tail "$tail_model" \
      --input "$input" \
      --output-dir "$directory" \
      --provider "$provider" \
      --cpu-list 0-3 \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup 0 \
      --runs 1 \
      --repeats 1 \
      "${profile_args[@]}"
  rc=$?
  set -e
  local output_hash=missing boundary_hash=missing profiles=0
  [[ -f $directory/output.bin ]] && output_hash=$(sha256sum "$directory/output.bin" | awk '{print $1}')
  if compgen -G "$directory/boundaries/*.bin" >/dev/null; then
    boundary_hash=$(sha256sum "$directory"/boundaries/*.bin | sha256sum | awk '{print $1}')
  fi
  profiles=$(find "$directory/profiles" -type f 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}_${provider}_${mode}" "$model" "$provider" "$mode" "$rc" \
    "$output_hash" "$boundary_hash" "$profiles" >>"$out/status.raw.tsv"
  [[ $rc -eq 0 ]] || return "$rc"
}

for model in B2 C2; do
  run_arm "$model" spacemit profile
  run_arm "$model" cpu fixture
  run_arm "$model" spacemit fixture
done

printf 'stage65d_r1_board_recheck status=pass\n'
