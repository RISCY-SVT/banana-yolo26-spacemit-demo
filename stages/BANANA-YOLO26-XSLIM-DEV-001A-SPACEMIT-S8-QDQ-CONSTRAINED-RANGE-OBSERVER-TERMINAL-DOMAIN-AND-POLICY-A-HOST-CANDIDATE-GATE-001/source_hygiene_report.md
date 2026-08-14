# Source hygiene report

## Scope

- XSlim source changes are limited to the authorized downstream branch and the
  constrained-range selector/search, S8-QDQ structural profile, tests, version metadata,
  notices, documentation, and compact source evidence.
- Banana changes contain deterministic host tooling, sanitized configs, compact reports,
  and manifests only.
- No new branch or tag was created. No release or PyPI publication occurred.
- No board command, custom-executor mutation, Policy B implementation, training, QAT,
  performance run, or runtime promotion occurred.

## Repository hygiene

- `git diff --check` and `git diff --cached --check`: pass before every commit.
- XSlim full pytest: 174 passed, 2 inherited warnings, 65 subtests.
- Focused selector/range/profile suite: 27 passed.
- Ruff, compileall, focused mypy, fresh wheel/sdist installs, `pip check`, and three CLI
  help smokes: pass.
- Whole-project mypy remains inherited debt: 971 errors across 69 files; no new focused
  error was introduced.
- No ONNX model, prediction JSON, dataset image/archive, activation tensor, bootstrap NPZ,
  wheel, sdist, cache, private key, token, Authorization header, or credential config is
  tracked in the Stage evidence.
- No symlink occurs in the tracked Stage evidence.
- XSlim `src/` contains no private YOLO tensor names.

## Raw evidence and interrupted work

Large models, predictions, bootstrap replicates, environments, and logs remain under the
authorized Stage raw root. Two bounded monolithic full-val bootstrap attempts were stopped
after runtime/robustness assessment; one additional retry failed closed on an existing
empty output directory. Their logs were preserved. The accepted resumable run reused the
exact frozen 1,000-replicate prefix and completed 10,000 replicates with checkpointing.
No bootstrap, PTQ, COCO, ONNX, pytest, or quantization process remained active at closure.

The accepted full-val replicate payload SHA-256 is
`f909a13ef8e09aa5c982f9fbe84f24bb7b19572e67fd2bb4dcffc589a7cdc765`;
the draw identity SHA-256 is
`4bd73bd3088da3d7ae85a0ef614ddc96c5ca009ac0b5c9431b85180e1554816b`.

Historical Stage65A build directories contain superseded wheel bytes. This Stage bound the
published release control only to the accepted PUB4 asset SHA-256
`635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784`.

## Export posture

Only compact reports, manifests, and small deterministic scripts are eligible for the
result packet. Raw evidence is referenced by path and hash. Export scanning is mandatory
before the official packet helper is run.
