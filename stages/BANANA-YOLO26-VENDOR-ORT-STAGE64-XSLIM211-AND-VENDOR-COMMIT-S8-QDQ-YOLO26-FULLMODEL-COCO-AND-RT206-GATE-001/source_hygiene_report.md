# Source hygiene

## Scope

Relative to accepted Stage63 head
`94b86a6bf011cc83fefaf2a960191e97a8daf728`, the final Stage64 branch changes
161 files:

- 25 files under `vendor_ort_validation/`;
- 136 files under the Stage64 report tree;
- zero files under `custom_int8_engine/`;
- zero files elsewhere.

Generated full YOLO models, calibration and holdout images, COCO predictions,
vendor runtimes, Python environments, profiles, faults, and large raw logs
remain outside Git under the task-local `/data` root.

## Checks

| Check | Result |
|---|---|
| `git diff --check` and cached check | pass |
| changed shell scripts, `bash -n` | pass |
| Python `compileall` | pass |
| Stage64 TSV rectangular-schema check | pass, 90 files |
| Stage64 JSON parse check | pass, 8 files |
| ShellCheck | unavailable; not reported as pass |
| Ruff | unavailable; not reported as pass |
| changed paths outside authorized trees | 0 |
| tracked symlinks in Stage64 scope | 0 |
| changed files larger than 5 MiB | 0 |
| tracked ONNX paths added by Stage64 | 0 |
| high-confidence secret/private-key scan | pass |
| protected settings/credential path scan | pass |
| custom executor source diff | empty |
| XSlim source-delta patch apply check | pass |
| public repro archive path safety | pass |
| public repro internal checksums | pass |
| public repro deterministic rebuild | byte-identical |

One generated Q/DQ schema table is 1,072,651 bytes. It is bounded,
machine-readable evidence; no other changed file exceeds 1 MiB. The
22,860-byte public repro archive contains only tiny synthetic ONNX controls,
their generated inputs and independent oracles, neutral runner source,
sanitized result tables, and the repository license. Its SHA-256 is
`c73f1807dfa2edfd3b82b2524fecbda004c9cac86af3824062d2498df16ab47d`.

`/data/ncnn` remains at
`a245a70c641a1f20f357c65d103e5f9e50fe84a1` with the same three pre-existing
modified convolution files and hashes. The protected custom-executor tree
remains `c2e400de14fb1c88d4aed70a249d9eff19a05d0f`. Four accepted release roots
retain exact before/after manifest identities.

Status: **pass with ShellCheck and Ruff unavailable**.
