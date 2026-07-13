# Source hygiene

- Start HEAD and clean initial worktree: pass.
- Host build and 48/48 CTests: pass.
- Correct host ASan/UBSan build and 48/48 CTests: pass.
- Python compile: pass.
- Full RISC-V cross-build and board loader: pass.
- RPATH/RUNPATH absolute build-tree paths: none.
- `git diff --check`: pass before report generation; repeated before commit.
- Symlink, large-file, scoped secret/private-path, and vendor/model/dataset exclusion scans: pass.
- `/data/ncnn`: unchanged from the Stage51 preflight baseline.
- Raw evidence root: `/data/ncnn-logs/ai-team/2026-07-13/2026-07-13_10-56-21__codex__BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE51-EXECUTOR-MAXIMIZATION-Q62-ISA-CLUSTER1-FULL-GRAPH-COVERAGE-NEXT-REGION-AND-PUBLISH-GATE-001__stage51-executor-max`.

One failed sanitizer attempt against a cross-configured build tree is preserved in raw evidence;
the corrected host ASan/UBSan run passed all tests. An intentionally broad first secret-pattern
scan matched the renderer's local variable `token`; the credential-specific rerun passed and the
false-positive attempt remains in the command ledger.
