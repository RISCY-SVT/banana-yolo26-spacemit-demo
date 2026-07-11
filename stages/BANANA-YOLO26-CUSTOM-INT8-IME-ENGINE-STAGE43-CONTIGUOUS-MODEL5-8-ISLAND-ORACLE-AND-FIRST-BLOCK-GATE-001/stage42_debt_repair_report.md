# Stage 42 Debt Repair

Stage 43 repaired only the bounded infrastructure items required by the task.

- The Stage 42 runner no longer includes `tests/test_stage16_model4_c2f_runner.cpp`. The shared model4 fixture factory now lives in `include/y26_k1x_model4_fixture_config.h` and `src/model4_fixture_config.cpp`.
- The Stage 42 runner target uses `SKIP_BUILD_RPATH`; host and RISC-V ELF inspection found no `RPATH` or `RUNPATH`.
- Scalar ONNX shape `[]` now has element count one. Focused tests cover null data, exact byte equality, signed histograms, percentile interpolation, finite float metrics, NaN/Inf policy, and invalid reference-policy parsing.
- Validation, benchmark, and profile modes remain separate. Detailed ORT session-init, first-run, input-wrap, `OrtRun`, and output-copy timings were added without putting reference work into the benchmark hot loop.
- Model5 code uses prepared weights, persistent workers, fixed workspaces, and no custom hot-loop allocation.

Historical tools that are not the Stage 42/43 runtime remain unchanged unless required for build compatibility.
