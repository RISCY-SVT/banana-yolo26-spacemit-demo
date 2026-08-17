# Source hygiene report

## Scope

- The Banana research branch contains only deterministic Stage65C validation
  tooling and compact reports/manifests.
- XSlim source, its development branch, release tag, and release artifacts are
  read-only in this Stage.
- A1, B2, datasets, predictions, vendor runtime binaries, profiles, cores, and
  bootstrap replicate payloads remain outside Git under the authorized raw root.
- No new branch, tag, release, PyPI publication, model generation, custom-engine
  execution, or default-runtime change occurred.

## Recovery and process hygiene

The interrupted concurrent bootstrap attempt was isolated under explicitly
named raw directories and is excluded from accepted results. Required bootstrap
pairs use new output directories and exact deterministic seeds. Closure requires
that no board runner, bootstrap, COCO evaluator, performance process, or soak
process remains active. The final process checks found none.

Accepted 10,000-replicate payload identities are:

- A1 EP versus B2 EP: `e59bb9544e23ddb120c5d278bc6eb9a67f523c888eb74483a52253457da9dca0`.
- A1 CPU versus B2 CPU: `e7bf3378cb143be18c3146649690a6bcab42a4563d5d1dfba004efd729315836`.
- A1 EP versus A1 CPU: `4255625a6f81b17a8f97c0b892dc079feb90376a01f6b31bd61652fa382fd446`.
- Shared deterministic draw matrix: `85dc0072d0baa6d17d1e94281b6b7e22f99c87faab00cdd9216b2aa9de6ae133`.

## Verification

- Ruff: pass for all `stage65c_*.py` tools.
- Compileall: pass.
- Shell syntax: pass for all four `stage65c_*.sh` tools.
- CLI help: 6/6 pass.
- Plugin report field-name uniqueness: pass after fixing a report-only duplicate
  `status` column; runtime evidence is unchanged.
- `git diff --check` and `git diff --cached --check`: pass.
- Secret/token-prefix, Authorization header, private-key material, raw credential
  path, symlink, hardlink, large-file, model/data/vendor-binary scans: pass.
- The tracked Stage tree contains no file larger than 5 MiB and no ONNX,
  prediction JSON, NPZ, shared library, archive, or dataset payload.

## Export posture

Only compact tracked reports, manifests, and small deterministic scripts are
eligible for export. The official bridge scanner and result-packet helper are
mandatory. Canonical raw paths are references only; model, prediction, dataset,
runtime-binary, and replicate bytes are not exportable.
