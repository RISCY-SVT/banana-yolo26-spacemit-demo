# Source hygiene report

## Scope

Only compact Stage65D tooling and reports were added to the existing Banana research branch. XSlim, the custom executor, `/data/ncnn`, frozen ONNX models, predictions, datasets and vendor runtime bytes were not modified or tracked.

## Validation

| Check | Result |
|---|---|
| Stage execution-environment `compileall` | pass |
| Every Stage65D Python CLI `--help` in the frozen environment | pass |
| Shell syntax (`bash -n`) | pass |
| JSON syntax | pass |
| `git diff --check` | pass |
| `git diff --cached --check` | pass before each commit |
| Secret/token/private-key pattern scan | pass |
| Symlink scan | pass; 0 links |
| Tracked-stage hardlink scan | pass; 0 multiply linked files |
| Files larger than 1 MiB | pass; 0 files |
| ELF/runtime-binary scan | pass; 0 files |
| Model/data/archive extension scan | pass; 0 forbidden payloads |
| Ruff | unavailable; recorded, not silently treated as pass |
| shellcheck | unavailable; recorded, not silently treated as pass |

The system Python environment does not include ONNX, but it was not used as the Stage execution environment. The frozen Stage environment imported ONNX and passed the execution-tool compile/help checks.

Raw model, prediction, bootstrap and provider-profile evidence remains under the Stage raw root and is referenced by SHA-256. It is excluded from Git and from the compact result packet.
