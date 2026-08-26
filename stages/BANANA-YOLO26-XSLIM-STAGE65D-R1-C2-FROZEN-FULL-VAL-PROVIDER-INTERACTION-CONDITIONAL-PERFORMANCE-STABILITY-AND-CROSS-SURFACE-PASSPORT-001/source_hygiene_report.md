# Source hygiene report

## Scope

Only compact Stage65D-R1 tooling and evidence were added to the existing Banana research branch. Frozen ONNX models, prediction JSON, bootstrap NPZ, COCO data, vendor binaries, board dumps, and large timing payloads remain outside Git. XSlim, the custom executor, protected Banana main, and `/data/ncnn` were not modified.

## Validation

| Check | Result |
|---|---|
| Stage execution-environment `compileall` | pass |
| Ten Stage65D-R1 Python CLIs, `--help` | pass |
| Four Stage65D-R1 shell scripts, `bash -n` | pass |
| Two JSON-subset YAML manifests, `jq` parse | pass |
| Native COCO metrics versus bootstrap point accumulator | pass; zero residual across 4 surfaces x 13 metrics |
| Score/rank complete-input reproducibility | pass; 6/6 reports byte-identical |
| Score/rank missing-interaction negative test | pass; fails closed before prediction loading |
| Performance/stability parser positive tests | pass |
| Corrupt sample/output hash negative tests | pass; fail closed |
| Custom-context parser contract test | pass |
| `git diff --check` | pass |
| Secret/token/private-key pattern scan | pass; zero findings |
| Symlink and multiply-linked-file scans | pass; zero findings |
| Files larger than 1 MiB in tracked Stage tree | pass; zero files |
| Model/data/runtime-binary extension scan | pass; zero forbidden payloads |
| Ruff | unavailable; not installed and not treated as pass |
| shellcheck | unavailable; not installed and not treated as pass |

Raw execution, prediction, bootstrap, and tooling-test evidence remains under the Stage raw root and is referenced by path and SHA-256. Conditional performance, soak, and custom execution files contain explicit `not-run-task-gate-closed` records; they are not empty evidence and are not reported as experimental failures.
