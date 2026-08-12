# Source hygiene report

Scope: changes from Stage65B-R2 start `40615a5d31ff1687d2f00b74f83470071793ab6c`.

- `git diff --check`: pass.
- Python unit tests: 4 passed, including the three required CSV field-limit
  cases and the vectorized bootstrap-envelope identity test.
- Python `compileall`: pass for every Stage65B-R2 module and test.
- Ruff: unavailable in the immutable released-XSlim environment; not installed
  or treated as a gate. No global environment was changed.
- Secret, token-prefix, Authorization-header, private-key-material, raw
  credential-path, and private-home-path scans: no findings.
- Symlink scan: no symlink in the Stage65B-R2 tracked tree.
- Large-file scan: largest new tracked file is the compact 520,976-byte
  normalized Graphwise TSV; no new tracked file exceeds 1 MiB.
- Model/data exclusion: no ONNX, JPEG, dataset archive, prediction JSON, NPY,
  or NPZ payload is tracked.
- Raw model, prediction, bootstrap, and activation evidence remains below the
  stage-local `/data/k1x-stage-runs` root and is referenced by hash.
- RPATH/RUNPATH scan: not applicable; no executable or shared-library payload
  was added.
- License/provenance scan: no vendored third-party source or binary was added;
  the new tooling uses repository and environment dependencies already covered
  by the existing research workflow.
- Board-scope scan: no board command, provider-placement claim, performance
  run, or runtime-promotion artifact was introduced.

Result: `pass`.
