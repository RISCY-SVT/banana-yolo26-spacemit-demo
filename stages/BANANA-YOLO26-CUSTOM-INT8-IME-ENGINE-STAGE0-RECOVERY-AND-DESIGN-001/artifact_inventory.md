# Artifact Inventory

Stage ID: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001`

## Scope Notes

- Primary repo: `/data/banana-yolo26-spacemit-demo`
- Reference old demo: `/data/banana-yolo11-spacemit-demo`
- `xslim` artifacts are excluded from authority for this YOLO26 custom engine path because the user reported a YOLO26 bug in `xslim`.
- No model was regenerated in Stage 0.

## Accepted IME Evidence Recovered

Accepted source paths:

- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/`
- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/2026-06-27_07-00-01_W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001-TIER2-REVIEW-AND-CLOSURE-001/accepted-facts.md`
- `/exchange/results/archive/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/`

Recovered accepted facts:

- `smt.vmadot`: integer IME primitive, accepted as `s8 x s8 -> s32`, 4x4x8 tile, cluster0 CPU0-3 pass.
- CPU4/CPU5: controlled `SIGILL` on the tested path.
- Exact scalar `int8 -> int32` oracle: accepted for six tested cases with zero mismatches.
- `vmadotn`: not proven; tested mnemonic rows rejected.
- `smt.vmadot3`: board execution proof exists, no independent oracle claim accepted for implementation.
- FP16 `vfmadot`: blocked/deferred.
- Implementation is not authorized by the proof alone.

## CPU-Good Q/DQ Candidate Set

Chosen candidate:

```text
/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
```

Alternative accepted/manual candidates:

| file | sha256 | size | mtime | output | CPU status |
| --- | --- | ---: | --- | --- | --- |
| `manual_e2e_rep_conv_matmul_qdq.onnx` | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` | 3072462 | 2026-06-29 17:07:21 +0200 | `[1,300,6]` | CPU-good, blank-clean |
| `manual_trad_rep_conv_matmul_qdq.onnx` | `c612fcbf5ce453198f75109d1907825a6d995a846bc0b5952f1b451323b14ca8` | 3044993 | 2026-06-29 17:07:51 +0200 | `[1,84,8400]` | CPU-good, blank-clean |
| `manual_e2e_rep_conv_only_qdq.onnx` | `535947b2ea4d03aa33af2a0f18759405bad9aaa6994bb729b23e715545f4cc8f` | 3063675 | 2026-06-29 17:07:31 +0200 | `[1,300,6]` | rejected: blank-image false positive |
| `manual_trad_rep_conv_only_qdq.onnx` | `8ee866e0b0612ce8bf0d3fa18feea3fda06a7b73b0b3305238a929c54483e0af` | 3036206 | 2026-06-29 17:08:02 +0200 | `[1,84,8400]` | CPU-good, blank-clean |

Common ONNX metadata for inspected candidates:

- opset: 18
- IR version: 8
- producer: `onnx.quantize`
- producer version: `0.1.0`
- model input: `images [1,3,640,640]`
- Q/DQ format: explicit `QuantizeLinear` / `DequantizeLinear`

Acceptance source:

- `docs/YOLO26_INT8_RT204_FORENSICS.md`
- `docs/YOLO26_ORACLE_RESULTS.md`
- raw evidence noted there under `/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/`, `/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/`, `/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/`

Reference images used by prior oracle claim:

- canonical production photo
- Ultralytics `bus.jpg`
- Ultralytics `zidane.jpg`
- Day 4 real camera still
- blank white sanity image

## Generated Stage 0 Evidence

- `onnx_stage0_digest.tsv`
- `matmul_quant_summary.tsv`
- `signedness_zero_point_audit.tsv`
- `$LOG_DIR/artifacts/onnx_graph_quant_summary.json`
- `$LOG_DIR/artifacts/first_matmul_shape_quant.json`
