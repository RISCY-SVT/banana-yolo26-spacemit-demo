#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  stage64_run_two_stage_board.sh \
    --runner FILE --runtime-lib DIR --inference FILE --tail FILE \
    --input FILE --output-dir DIR --provider cpu|spacemit \
    [--cpu-list 0-3] [--intra-threads 4] [--inter-threads 1] \
    [--warmup 10] [--runs 100] [--repeats 5]
EOF
}

runner=
runtime_lib=
inference=
tail=
input=
output_dir=
provider=
cpu_list=0-3
intra_threads=4
inter_threads=1
warmup=10
runs=100
repeats=5

while (($#)); do
  case "$1" in
    --runner) runner=${2:?}; shift 2 ;;
    --runtime-lib) runtime_lib=${2:?}; shift 2 ;;
    --inference) inference=${2:?}; shift 2 ;;
    --tail) tail=${2:?}; shift 2 ;;
    --input) input=${2:?}; shift 2 ;;
    --output-dir) output_dir=${2:?}; shift 2 ;;
    --provider) provider=${2:?}; shift 2 ;;
    --cpu-list) cpu_list=${2:?}; shift 2 ;;
    --intra-threads) intra_threads=${2:?}; shift 2 ;;
    --inter-threads) inter_threads=${2:?}; shift 2 ;;
    --warmup) warmup=${2:?}; shift 2 ;;
    --runs) runs=${2:?}; shift 2 ;;
    --repeats) repeats=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in runner runtime_lib inference tail input output_dir provider; do
  if [[ -z ${!required} ]]; then
    printf 'missing --%s\n' "${required//_/-}" >&2
    usage >&2
    exit 2
  fi
done
if [[ $provider != cpu && $provider != spacemit ]]; then
  printf 'invalid provider: %s\n' "$provider" >&2
  exit 2
fi
if [[ $output_dir != /data/* ]]; then
  printf 'output directory must be on /data: %s\n' "$output_dir" >&2
  exit 2
fi
for file in "$runner" "$inference" "$tail" "$input"; do
  [[ -f $file ]] || { printf 'missing file: %s\n' "$file" >&2; exit 2; }
done
[[ -d $runtime_lib ]] || {
  printf 'missing runtime library directory: %s\n' "$runtime_lib" >&2
  exit 2
}

mkdir -p \
  "$output_dir/boundaries" \
  "$output_dir/profiles" \
  "$output_dir/tmp" \
  "$output_dir/cache"
cat /proc/sys/kernel/random/boot_id >"$output_dir/boot_id.txt"
find /sys/devices/system/cpu/cpufreq -name scaling_cur_freq -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/frequency_before.tsv"
find /sys/class/thermal -name temp -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/thermal_before.tsv"

command=(
  "$runner"
  --provider "$provider"
  --inference-model "$inference"
  --tail-model "$tail"
  --input "$input"
  --output "$output_dir/output.bin"
  --boundary-output-dir "$output_dir/boundaries"
  --profile-prefix "$output_dir/profiles/profile"
  --samples-output "$output_dir/samples.tsv"
  --intra-threads "$intra_threads"
  --inter-threads "$inter_threads"
  --warmup "$warmup"
  --runs "$runs"
  --repeats "$repeats"
)

{
  printf 'provider=%s\n' "$provider"
  printf 'cpu_list=%s\n' "$cpu_list"
  printf 'runtime_lib=%s\n' "$runtime_lib"
  printf 'inference=%s\n' "$inference"
  printf 'tail=%s\n' "$tail"
  printf 'input=%s\n' "$input"
  printf 'warmup=%s\nruns=%s\nrepeats=%s\n' "$warmup" "$runs" "$repeats"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$output_dir/effective-config.txt"

(
  cd "$output_dir"
  env \
    LD_LIBRARY_PATH="$runtime_lib" \
    TMPDIR="$output_dir/tmp" \
    XDG_CACHE_HOME="$output_dir/cache" \
    /usr/bin/time -v -o "$output_dir/time-v.txt" \
    taskset -c "$cpu_list" \
    "${command[@]}"
) >"$output_dir/run.log" 2>&1

sha256sum "$output_dir/output.bin" >"$output_dir/output.sha256"
sha256sum "$output_dir"/boundaries/*.bin >"$output_dir/boundary-sha256.txt"
find /sys/devices/system/cpu/cpufreq -name scaling_cur_freq -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/frequency_after.tsv"
find /sys/class/thermal -name temp -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/thermal_after.tsv"
