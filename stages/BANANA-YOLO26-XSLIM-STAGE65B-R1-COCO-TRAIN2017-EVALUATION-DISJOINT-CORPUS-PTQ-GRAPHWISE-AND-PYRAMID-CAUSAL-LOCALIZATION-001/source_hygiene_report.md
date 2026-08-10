# Source hygiene report

Status: `pass`.

The tracked scope contains Stage65B-R1 host tooling, compact evidence tables,
effective configurations, and reports. Dataset archives and JPEGs, generated
ONNX models, prediction JSON, caches, Python environments, and raw Graphwise
tensors remain outside Git under `/data` and were not synchronized to Drive.

Checks completed:

- secret, token-prefix, Authorization-header, private-key, and credential-path
  scans: pass with zero findings;
- private-path scan: pass; explicit `/data` raw-evidence references are part of
  the lab evidence contract, while credential and Codex configuration roots are
  absent;
- symlink and hardlink scans: zero findings;
- model, weight, image, archive, NumPy payload, and prediction-JSON exclusion:
  pass;
- files over 10 MiB: zero; the largest tracked report is the required
  5,171,486-byte boundary-saturation table containing per-image hashes;
- repository AGPL-3.0 license and existing license inventory remain present;
  no third-party source was copied into the new tooling;
- `compileall`, explicit `py_compile`, all Stage65B-R1 CLI help smokes,
  `pip check`, large-TSV regression, `git diff --check`, and cached diff check:
  pass;
- Ruff: not run because the executable is unavailable in the immutable stage
  environment; this is recorded as a non-gating tool absence.

The full matrix itself is the end-to-end integration test: two clean seeded
PTQ generations for every B1-B6 lane, host graph and semantic gates, Graphwise
and boundary audits, six hybrid arms, and all required 5000-image COCO runs
completed. Final process inspection found no surviving stage process.
