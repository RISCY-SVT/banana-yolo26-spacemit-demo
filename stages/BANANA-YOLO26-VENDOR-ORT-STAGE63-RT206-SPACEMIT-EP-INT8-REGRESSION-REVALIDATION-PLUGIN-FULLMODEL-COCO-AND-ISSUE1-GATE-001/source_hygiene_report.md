# Source hygiene

## Scope

Relative to protected main
`1fd2e71bb1d5a924e7c0444cada94f681b73aa91`, the Stage63 content commit changes
99 files:

- 16 files under `vendor_ort_validation/`;
- 83 files under the Stage63 report tree;
- zero files under `custom_int8_engine/`;
- zero files elsewhere.

## Checks

| Check | Result |
|---|---|
| `git diff --check` | pass |
| changed shell scripts, `bash -n` | pass |
| Python `compileall` | pass |
| task helper CLI smoke | pass with recorded task-local environments |
| ShellCheck | unavailable; not reported as pass |
| tracked symlinks in Stage63 scope | 0 |
| newly changed files larger than 1 MiB | 0 |
| secret/private-key assignment scan | pass |
| protected Codex/settings/credential path scan | pass |
| custom executor source diff | empty |
| public repro bundle internal checksums | pass |
| public repro bundle escaping symlinks | 0 |
| public repro bundle files larger than 1 MiB | 0 |
| public repro bundle deterministic rebuild | byte-identical |

The repository already contains 12 blobs larger than 1 MiB outside this
Stage63 delta. None was added or modified by this branch.

The minimal issue bundle contains only tiny synthetic ONNX models, inputs,
independent expected values, generic runner/plugin sources, the decision table,
and the repository license. It contains no full YOLO model, trained weights,
COCO data, camera media, vendor runtime binary, credential, or private machine
path.

`/data/ncnn` remains at
`a245a70c641a1f20f357c65d103e5f9e50fe84a1` with the same three pre-existing
modified convolution files recorded at Stage63 preflight. Four protected
release roots retain exact before/after manifest identities.

Status: **pass with ShellCheck unavailable**.
