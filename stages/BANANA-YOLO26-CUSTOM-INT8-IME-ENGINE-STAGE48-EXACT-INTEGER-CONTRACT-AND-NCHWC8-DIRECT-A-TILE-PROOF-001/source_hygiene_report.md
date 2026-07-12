# Source hygiene report

- `git diff --check`: pass.
- `git diff --cached --check`: pass before staging.
- Symlink scan under `custom_int8_engine` and `stages`: zero results.
- Largest changed/untracked file: 41,789 bytes; no model, binary, build tree,
  tensor dump, dataset, vendor runtime, or raw board log is in the repository set.
- Secret-like scan: zero results.
- Private credential/config path scan: zero results.
- Host build and 47-test CTest suite: pass.
- Focused x86 ASan/UBSan Stage48 tests: 2/2 pass. The first sanitizer configure
  accidentally inherited the RISC-V compiler and is preserved as a failed
  harness attempt; the explicit `/usr/bin/g++` rebuild is the accepted result.
- Python compile: pass.
- Deterministic package regeneration: pass; complete directory diff is empty.
- Full RISC-V cross-build: pass. Historical unrelated tools still emit existing
  unused-function warnings from direct test-source inclusion; Stage48 adds no
  such inclusion.
- Board loader: pass with no RPATH/RUNPATH and only system runtime dependencies.
- `/data/ncnn`: not mutated by Stage48. Its pre-existing dirty paths and all
  three SHA-256 values match the recorded start state.
- Push: forbidden and not performed.
