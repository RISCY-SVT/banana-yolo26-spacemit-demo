# Legacy Runtime Q/DQ Sanity Gate

Run directory:

```text
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/
```

## Question

Could older SpacemiT ORT runtimes execute the CPU-good YOLO26 manual Q/DQ INT8
candidates better than `rt204`?

## Runtimes Tested

| Tag | Runtime bundle | Result |
| --- | --- | --- |
| `rt123` | `spacemit-ort.riscv64.1.2.3` | Minimal repros run, but full Q/DQ outputs do not match same-runtime CPU and no SpaceMIT subgraph dump was produced for the accepted-looking full row. Not accepted. |
| `rt201` | `spacemit-ort.riscv64.2.0.1` | Reproduces `output_type not implemented for clip minmax` on Q/DQ Conv. Filter fallback runs but disables Q/DQ/Conv and does not match CPU hash. |
| `rt202b1` | `spacemit-ort.riscv64.2.0.2+beta1` | Reproduces `clip minmax` on default Q/DQ Conv. Diagnostic filter rows can run, but do not preserve CPU hash and are fallback-only. |
| `rt202` | `spacemit-ort.riscv64.2.0.2` | Fails this gate with `tcm buffer alloc failed for core id 0` on SpaceMIT EP rows. |
| `rt204` | `spacemit-ort.riscv64.2.0.4` | Control runtime; reproduces the known `clip minmax` Q/DQ Conv blocker. |

## Candidates

| Candidate | Purpose | Result |
| --- | --- | --- |
| `15_conv_qdq_attr_kernel_shape.onnx` | Tiny synthetic Q/DQ Conv repro. | CPU passes on all runtimes. `rt201`, `rt202b1`, and `rt204` fail SpaceMIT EP with `clip minmax`; `rt202` fails with TCM allocation; `rt123` runs but does not prove a full-model path. |
| `07_yolo26_first_conv_qdq_output_block.onnx` | Small real YOLO26-derived repro. | Same pattern as the tiny repro. |
| `manual_e2e_rep_conv_matmul_qdq.onnx` | CPU-good full YOLO26 e2e manual Q/DQ candidate. | Default SpaceMIT EP fails with `clip minmax` on `rt201`, `rt202b1`, and `rt204`; `rt202` fails TCM allocation; `rt123` runs but output hash differs from same-runtime CPU. |
| `manual_trad_rep_conv_matmul_qdq.onnx` | CPU-good full YOLO26 traditional manual Q/DQ candidate. | Same default failure class as e2e on `rt201`, `rt202b1`, and `rt204`; `rt123` output differs from same-runtime CPU. |
| `e2e_qdq_rep_conv_matmul_strip_kernel_shape.onnx` | Diagnostic stripped-kernel candidate. | Still not accepted: default rows fail or mismatch, and filter rows are partial fallbacks. |

## Placement Findings

The only rows with dumped SpaceMIT subgraphs were diagnostic filter rows. They
are not accepted as accelerated INT8:

- `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`
  moves Q/DQ Conv away from SpaceMIT EP. The dumped subgraphs contain MatMul and
  head fragments, not quantized Conv offload.
- `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` on the stripped-kernel
  diagnostic model dumps Q/DQ fragments but disables the attention operations
  needed to avoid the second blocker. Output hashes still differ from CPU.
- `rt123` full-model rows produced no SpaceMIT subgraph dump and did not match
  same-runtime CPU hashes, so they are not evidence of a correct accelerated
  path.

No row satisfied all acceptance criteria:

```text
CPU oracle good
SpaceMIT EP run succeeds
same-runtime CPU parity preserved
meaningful INT8 compute offload visible
```

## Decision

```text
LEGACY_RUNTIME_QDQ_CLOSED_NO_ACCELERATED_PATH: yes
```

The YOLO26 INT8 closure does not change. Older SpacemiT ORT runtimes do not
provide an accepted accelerated Q/DQ INT8 path for the current CPU-good YOLO26
manual Q/DQ candidates.

## Evidence

Key task tables:

```text
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/runtime_inventory.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/qdq_candidate_inventory.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/cpu_oracle_revalidation.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/legacy_runtime_minimal_repro_matrix.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/legacy_runtime_full_qdq_matrix.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/tables/legacy_runtime_full_qdq_parity.md
/data/ncnn-logs/ort-logs/2026-06-30_17-29-25/artifacts/legacy_runtime_placement_report.md
```
