# Source Hygiene Report

## Scope

The candidate commit is limited to the inherited Stage41 scaffold/reports, Stage40 traceability corrections, Stage42 support/runtime tooling, tests, and Stage42 reports. Build trees, tensor dumps, binaries, ORT profiles, credentials, and raw board artifacts remain outside Git.

## Checks

- `git diff --check`: pass.
- changed/untracked symlink scan under `custom_int8_engine` and `stages`: 0 symlinks.
- changed/untracked files larger than 1 MiB: none.
- secret-pattern scan: `rg` exit 1, expected no matches.
- private-path scan for Codex state, Settings, secrets, and SSH roots: `rg` exit 1, expected no matches.
- Python compile: pass.
- host build and CTest: 43/43 pass.
- RISC-V cross build with IME enabled: pass.
- board loader: intended ORT library resolved.
- board scalar/IME same-input controls: pass.
- CPU4-7 IME execution: none launched.

## External trees

`/data/ncnn` has unrelated pre-existing modifications in three `convolution_1x1_int8_xsmtvdot` files. Their mtimes are 2026-05-27, before this run, and no Stage42 command modified that tree. `/data/banana-yolo11-spacemit-demo` remained at `e0f091ca875572185fbd835e49e7d14aab559e1b` and was not modified.

## Reviewed limitations

The final cross runner still carries an absolute build-tree RPATH, documented as non-deployable technical debt. The Stage41 tool's test-fixture source inclusion is also documented and deferred; neither issue changes Stage42 numerical results.
