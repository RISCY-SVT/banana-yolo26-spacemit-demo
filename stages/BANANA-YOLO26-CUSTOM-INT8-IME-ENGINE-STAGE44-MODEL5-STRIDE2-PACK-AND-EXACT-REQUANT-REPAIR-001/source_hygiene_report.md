# Source hygiene

Final pre-commit status: pass.

- `/data/ncnn`: read-only inspection only; no Stage44 mutation.
- `/data/banana-yolo11-spacemit-demo`: not modified.
- Build trees, ONNX models, tensor dumps, board outputs, logs, credentials, and machine-private files are excluded from Git.
- New runtime modes are explicit/non-default.
- No absolute build-tree RPATH/RUNPATH is present in the board runner; `ldd` resolves the intended board ORT library through explicit `LD_LIBRARY_PATH`.
- Secret/private-path scan scope: changed and untracked repository files.
- Symlink scan scope: `custom_int8_engine` and Stage44 reports.
- Large-file scan scope: staged files before commit.

Observed results:

- `git diff --check`: pass.
- Python compile: pass.
- Host build and CTest: pass, 44/44 tests.
- Focused ASan/UBSan test: pass.
- RISC-V IME cross-build: pass.
- Symlinks: none.
- Files larger than 5 MiB in the changed source/report scope: none.
- Secret/private-key patterns in changed and untracked files: none.
- Absolute build-tree RPATH/RUNPATH: none.
- `/data/ncnn` working tree remained unchanged by Stage44.

Raw command/test evidence is under the Stage44 log root and is referenced by the result packet, not copied into Git.
