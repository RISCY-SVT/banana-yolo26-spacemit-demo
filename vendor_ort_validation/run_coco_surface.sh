#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_coco_surface.sh \
    --predictor FILE --runtime DIR --opencv DIR --model FILE \
    --labels FILE --images DIR --output-dir DIR \
    --surface NAME --provider cpu|spacemit [--limit N]

Runs one isolated YOLO26 COCO prediction surface. The runtime directory must
contain lib/libonnxruntime.so and lib/libspacemit_ep.so.
EOF
}

predictor=
runtime=
opencv=
model=
labels=
images=
output_dir=
surface=
provider=
limit=0

while (($#)); do
    case "$1" in
        --predictor) predictor=${2:?}; shift 2 ;;
        --runtime) runtime=${2:?}; shift 2 ;;
        --opencv) opencv=${2:?}; shift 2 ;;
        --model) model=${2:?}; shift 2 ;;
        --labels) labels=${2:?}; shift 2 ;;
        --images) images=${2:?}; shift 2 ;;
        --output-dir) output_dir=${2:?}; shift 2 ;;
        --surface) surface=${2:?}; shift 2 ;;
        --provider) provider=${2:?}; shift 2 ;;
        --limit) limit=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in predictor runtime opencv model labels images output_dir surface provider; do
    if [[ -z ${!value} ]]; then
        printf 'missing --%s\n' "${value//_/-}" >&2
        exit 2
    fi
done

[[ -x $predictor ]] || { printf 'predictor is not executable: %s\n' "$predictor" >&2; exit 2; }
[[ -f $runtime/lib/libonnxruntime.so ]] || { printf 'missing runtime core under %s\n' "$runtime" >&2; exit 2; }
[[ -f $runtime/lib/libspacemit_ep.so ]] || { printf 'missing provider under %s\n' "$runtime" >&2; exit 2; }
[[ -d $opencv/lib ]] || { printf 'missing OpenCV lib directory: %s\n' "$opencv" >&2; exit 2; }
[[ -f $model ]] || { printf 'missing model: %s\n' "$model" >&2; exit 2; }
[[ -f $labels ]] || { printf 'missing labels: %s\n' "$labels" >&2; exit 2; }
[[ -d $images ]] || { printf 'missing image directory: %s\n' "$images" >&2; exit 2; }
[[ $provider == cpu || $provider == spacemit ]] || { printf 'invalid provider: %s\n' "$provider" >&2; exit 2; }
[[ $limit =~ ^[0-9]+$ ]] || { printf 'invalid limit: %s\n' "$limit" >&2; exit 2; }

mkdir -p "$output_dir"
predictions="$output_dir/${surface}.predictions.json"
timing="$output_dir/${surface}.timing.tsv"
stdout="$output_dir/${surface}.stdout"
stderr="$output_dir/${surface}.stderr"
identity="$output_dir/${surface}.identity.tsv"

{
    printf 'field\tvalue\n'
    printf 'surface\t%s\n' "$surface"
    printf 'provider\t%s\n' "$provider"
    printf 'predictor\t%s\n' "$predictor"
    printf 'predictor_sha256\t%s\n' "$(sha256sum "$predictor" | awk '{print $1}')"
    printf 'runtime\t%s\n' "$runtime"
    printf 'onnxruntime_sha256\t%s\n' "$(sha256sum "$runtime/lib/libonnxruntime.so" | awk '{print $1}')"
    printf 'spacemit_ep_sha256\t%s\n' "$(sha256sum "$runtime/lib/libspacemit_ep.so" | awk '{print $1}')"
    printf 'model\t%s\n' "$model"
    printf 'model_sha256\t%s\n' "$(sha256sum "$model" | awk '{print $1}')"
    printf 'images\t%s\n' "$images"
    printf 'limit\t%s\n' "$limit"
    printf 'affinity\t0-3\n'
    printf 'threads\t4\n'
    printf 'graph_optimization\tall\n'
} >"$identity"

export LD_LIBRARY_PATH="$runtime/lib:$opencv/lib"
export TMPDIR="${TMPDIR:-$output_dir/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$output_dir/cache}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset SPACEMIT_EP_DUMP_SUBGRAPHS
cd "$output_dir"

set +e
timeout_seconds=${STAGE63_COCO_TIMEOUT_SECONDS:-14400}
timeout --signal=TERM --kill-after=30s "${timeout_seconds}s" \
  taskset -c 0-3 "$predictor" \
    --model "$model" \
    --labels "$labels" \
    --images "$images" \
    --output "$predictions" \
    --timing-tsv "$timing" \
    --provider "$provider" \
    --threads 4 \
    --pin cluster0 \
    --input-size 640 \
    --conf 0.001 \
    --iou 0.7 \
    --limit "$limit" \
    --log-every 100 >"$stdout" 2>"$stderr"
rc=$?
set -e

{
    printf 'exit_code\t%s\n' "$rc"
    if [[ -f $predictions ]]; then
        printf 'predictions_sha256\t%s\n' "$(sha256sum "$predictions" | awk '{print $1}')"
    fi
    if [[ -f $timing ]]; then
        printf 'timing_sha256\t%s\n' "$(sha256sum "$timing" | awk '{print $1}')"
    fi
} >>"$identity"

exit "$rc"
