# Source hygiene report

## Scope

- The Banana branch contains small deterministic Stage65C-R1 tooling and
  compact reports/manifests only.
- A1/B2 ONNX, predictions, COCO images, boundary arrays, bootstrap NPZ, vendor
  binaries, profiles, and recovery payloads remain under the authorized raw
  root and are not tracked.
- XSlim, custom executor, ncnn, published refs/releases, and default runtime
  were not mutated.
- No new branch, tag, release, model, PTQ policy, or later-stage prompt was
  created.

## Recovery and process hygiene

The incomplete pre-restart smoke and the clean `awk` reproduction are isolated
under named raw recovery roots and excluded from accepted results. All accepted
runs use fresh fail-on-existing directories. Final process inspection found no
Stage runner, ORT inference, SSH runner, COCO evaluator, bootstrap, performance,
or soak process.

## Verification

- Ruff: pass for all eight `stage65c_r1_*.py` files.
- Compileall: pass for all eight Python files.
- Shell syntax: pass for all five `stage65c_r1_*.sh` files.
- CLI help: 8/8 pass using dependency-correct COCO and ONNX/ORT environments.
- Shellcheck: unavailable; `bash -n` is the recorded shell gate.
- Board deterministic matrix: all eight model/provider surfaces pass one
  100-run session and 10 clean session recreations.
- Full val bootstrap: 10,000 shared draws; output/checksum contract pass.
- `git diff --check` and `git diff --cached --check`: pass.
- Secret/token-prefix, Authorization header, private-key material, raw
  credential path, symlink/hardlink, large-file, model/data/runtime-binary, and
  accidental executable scans: pass.
- No tracked Stage file exceeds 5 MiB; no ONNX, prediction JSON, NPZ, shared
  library, archive, dataset image, or raw boundary tensor is tracked.

## Export posture

Only compact tracked reports and manifests are exportable. Raw evidence is
referenced by canonical `/data` paths and SHA-256 identities. Export uses the
bridge candidate scanner and `/data/lab/scripts/export-result-packet.sh`; no raw
payload is copied into `/exchange`.
