#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65D_BOARD_ROOT:?STAGE65D_BOARD_ROOT is required}"
root=$STAGE65D_BOARD_ROOT
out="$root/determinism"
runner="$root/bin/stage64_two_stage_runner"
wrapper="$root/bin/stage65d_run_two_stage_board.sh"
tail_model="$root/models/stage65b_r1_b2.postprocess.onnx"
input="$root/fixtures/fixed/images_F0_f32.bin"
[[ ! -e $out ]] || { printf 'determinism output exists: %s\n' "$out" >&2; exit 2; }
mkdir -p "$out"
printf 'model\tprovider\tmode\trecreation\truns\texit_code\toutput_sha256\tboundary_manifest_sha256\tunique_sample_hashes\tsamples_sha256\tstatus\n' >"$out/status.raw.tsv"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    C2) printf '%s' "$root/models/c2_t6_rank_qp.inference.onnx" ;;
    *) return 2 ;;
  esac
}

manifest_sha() {
  local directory=$1
  for path in "$directory"/boundary-*.bin; do
    printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$(basename "$path")"
  done | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

run_one() {
  local model=$1 provider=$2 mode=$3 recreation=$4 warmup=$5 runs=$6
  local output="$out/${model}-${provider}/${mode}-${recreation}"
  local inference rc output_sha boundary_sha samples_sha unique_hashes
  inference=$(inference_for "$model")
  [[ ! -e $output ]]
  rc=0
  timeout --signal=TERM --kill-after=20s 1800s \
    "$wrapper" \
      --runner "$runner" \
      --runtime-lib "$root/runtime/lib" \
      --inference "$inference" \
      --tail "$tail_model" \
      --input "$input" \
      --output-dir "$output" \
      --provider "$provider" \
      --cpu-list 0-3 \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup "$warmup" \
      --runs "$runs" \
      --repeats 1 || rc=$?
  [[ $rc -eq 0 ]]
  output_sha=$(sha256sum "$output/output.bin" | awk '{print $1}')
  boundary_sha=$(manifest_sha "$output/boundaries")
  samples_sha=$(sha256sum "$output/samples.tsv" | awk '{print $1}')
  unique_hashes=$(tail -n +2 "$output/samples.tsv" | cut -f6 | LC_ALL=C sort -u | wc -l)
  [[ $unique_hashes -eq 1 ]]
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tpass\n' \
    "$model" "$provider" "$mode" "$recreation" "$runs" "$rc" \
    "$output_sha" "$boundary_sha" "$unique_hashes" "$samples_sha" >>"$out/status.raw.tsv"
}

for model in B2 C2; do
  for provider in cpu spacemit; do
    run_one "$model" "$provider" one-session 0 2 100
    for recreation in $(seq 1 10); do
      run_one "$model" "$provider" recreate "$recreation" 0 1
    done
    output_count=$(awk -F '\t' -v m="$model" -v p="$provider" 'NR>1 && $1==m && $2==p {print $7}' "$out/status.raw.tsv" | sort -u | wc -l)
    boundary_count=$(awk -F '\t' -v m="$model" -v p="$provider" 'NR>1 && $1==m && $2==p {print $8}' "$out/status.raw.tsv" | sort -u | wc -l)
    [[ $output_count -eq 1 && $boundary_count -eq 1 ]]
  done
done

printf 'stage65d_board_determinism status=pass\n'
