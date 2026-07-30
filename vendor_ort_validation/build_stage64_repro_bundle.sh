#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  build_stage64_repro_bundle.sh \
    --stage-root DIR \
    --repo-root DIR \
    --report-root DIR \
    --output FILE.tar.gz

Build a deterministic public-safe Stage64 bundle containing only synthetic
tiny ONNX controls, their independent inputs/oracles, a neutral runner, and
sanitized result tables.
EOF
}

stage_root=
repo_root=
report_root=
output=

while (($#)); do
    case "$1" in
        --stage-root) stage_root=${2:?}; shift 2 ;;
        --repo-root) repo_root=${2:?}; shift 2 ;;
        --report-root) report_root=${2:?}; shift 2 ;;
        --output) output=${2:?}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for variable in stage_root repo_root report_root output; do
    test -n "${!variable}" || {
        printf 'missing --%s\n' "${variable//_/-}" >&2
        exit 2
    }
done

stage_root=$(realpath "$stage_root")
repo_root=$(realpath "$repo_root")
report_root=$(realpath "$report_root")
output=$(realpath -m "$output")

model_root="${stage_root}/models/tiny"
oracle_root="${stage_root}/host/tiny-oracles"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bundle="$tmp/ISSUE_1_STAGE64_MINIMAL_S8_QDQ_REPRO_BUNDLE"
mkdir -p \
    "$bundle/models" \
    "$bundle/inputs" \
    "$bundle/expected" \
    "$bundle/runner" \
    "$bundle/results"

for model in \
    c1_s8_conv_pc_explicit \
    c2_s8_conv_pc_nonzero_zp \
    c3_s8_conv_pc_no_kernel \
    c4_u8_conv_pc_explicit \
    c5_s8_conv_pt_explicit \
    m1_s8_matmul \
    m2_s8_matmul_nonzero_zp \
    m3_u8_matmul \
    reducemax_two_input_opset18 \
    reducemax_attr_opset13 \
    conv_reducemax_two_input_opset18; do
    install -m 0644 "${model_root}/${model}.onnx" "$bundle/models/"
    install -m 0644 "${oracle_root}/${model}.input.bin" "$bundle/inputs/"
    install -m 0644 "${oracle_root}/${model}.oracle.bin" "$bundle/expected/"
done

install -m 0644 \
    "${repo_root}/vendor_ort_validation/stage64_single_model_runner.cpp" \
    "$bundle/runner/"
install -m 0644 \
    "${repo_root}/vendor_ort_validation/stage64_tiny_oracles.py" \
    "$bundle/runner/"
install -m 0644 "${repo_root}/LICENSE" "$bundle/LICENSE"
install -m 0644 \
    "${report_root}/vendor_claim_evidence_matrix.tsv" \
    "$bundle/results/"

cut -f1-18 "${report_root}/tiny_vendor_contract_matrix.tsv" \
    >"$bundle/results/tiny_vendor_contract_matrix.tsv"
cut -f1-12 "${report_root}/reducemax_regression_matrix.tsv" \
    >"$bundle/results/reducemax_regression_matrix.tsv"

cat >"$bundle/README.md" <<'EOF'
# Stage64 signed-INT8 QDQ minimal controls

This public-safe bundle contains only tiny generated ONNX graphs, synthetic
inputs, independent expected outputs, and a neutral ONNX Runtime runner. It
contains no YOLO model, trained weights, calibration or COCO data, camera
media, custom-executor source, vendor binary, credential, or private path.

## Cases

- `c1`, `c2`, `c5`: signed-INT8 QDQ Conv controls with explicit
  `kernel_shape`, including zero/nonzero activation zero points and
  per-channel/per-tensor weights.
- `c3`: signed-INT8 QDQ Conv without `kernel_shape`; exact CPU fallback is the
  placement control.
- `c4`: UINT8 QDQ Conv negative control.
- `m1`, `m2`: signed-INT8 QDQ MatMul controls with zero/nonzero activation
  zero points.
- `m3`: UINT8 QDQ MatMul negative control.
- `reducemax*`: XSlim 2.1.1 versus vendor-reference parser controls.

The files under `expected/` come from an independent local NumPy/operator
oracle, not from SpacemiT EP output.

## Build the neutral runner

Set `CXX` to the RISC-V compiler and `ORT_ROOT` to the extracted official
SpacemiT ORT 2.0.6 package:

```bash
"${CXX}" -std=c++17 -O2 -DNDEBUG -march=rv64gc -mabi=lp64d \
  -I"${ORT_ROOT}/include" \
  runner/stage64_single_model_runner.cpp \
  -L"${ORT_ROOT}/lib" -lspacemit_ep -lonnxruntime -ldl -pthread \
  -Wl,--no-undefined -o stage64_single_model_runner
```

Run CPU and provider arms independently:

```bash
LD_LIBRARY_PATH="${ORT_ROOT}/lib" taskset -c 0 timeout 90s \
  ./stage64_single_model_runner \
    --provider spacemit \
    --model models/c1_s8_conv_pc_explicit.onnx \
    --input inputs/c1_s8_conv_pc_explicit.input.bin \
    --output c1.output.bin

cmp c1.output.bin expected/c1_s8_conv_pc_explicit.oracle.bin
```

Use a process boundary and capture exit status/signal for negative controls.
The Stage64 evidence records that signed controls execute exactly, omitted
`kernel_shape` falls back to CPU, and UINT8 controls report unsupported
quantization before terminating with SIGABRT.

The bundle is evidence only. It does not promote a runtime or make a
production claim.
EOF

(
    cd "$bundle"
    find . -type f ! -name SHA256SUMS -print0 |
        sort -z |
        xargs -0 sha256sum >SHA256SUMS
)

mkdir -p "$(dirname "$output")"
epoch=${SOURCE_DATE_EPOCH:-0}
tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
    -C "$tmp" -cf - ISSUE_1_STAGE64_MINIMAL_S8_QDQ_REPRO_BUNDLE |
    gzip -n >"$output"
(
    cd "$(dirname "$output")"
    sha256sum "$(basename "$output")" \
        >"$(basename "${output%.tar.gz}.sha256")"
)
