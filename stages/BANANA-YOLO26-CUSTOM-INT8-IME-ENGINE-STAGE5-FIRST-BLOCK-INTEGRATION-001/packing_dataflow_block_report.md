# Packing Dataflow Block Report

Stage 5 integrates Stage 4 persistent packing into a minimal first-block runner.

## Dataflow

Prepare phase:

1. Validate selected block config.
2. Prepack real `/model.0/conv/Conv` weights once with `Y26PrepackedConvWeights`.
3. Allocate reusable `Y26ConvWorkspace`.
4. Allocate aligned raw int32 scratch for selected block output.

Run phase:

1. Pack activation panels through the Stage 4 M-major prepacked Conv path.
2. Execute plain `smt.vmadot` MMT4D `4x4x8 s8xs8->s32` tiles on cluster0.
3. Apply selected-node zero-point correction into caller-provided corrected int32 output.

No heap allocation is required inside `y26_stage5_block0_run_scalar` or `y26_stage5_block0_run_ime_cluster0_hotpath` after `prepare`.

## Memory

Small correctness ROI:

- prepacked bytes: `576`
- workspace bytes: `128`
- raw int32 scratch bytes: `1024`

Full selected block microbench:

- prepacked bytes: `576`
- workspace bytes: `128`
- raw int32 scratch bytes: `6553600`

## Timing

On board CPU0, `bench_stage5_first_block 3`:

- packA probe: `38912.7 us`
- IME total packing included: `71932.7 us`
- residual compute/correction estimate: `33020 us`

The selected block is faster than scalar, but packA remains a major component and should stay visible in Stage 6 reports.
