# Stage31 Decision

classification: stage31-direct-conv-correct-but-no-speed-win

Decision:

The Stage30-proven `smt.vmadot1/2/3` instructions were successfully replayed and used in a real-node direct/sliding 3x3 Conv sidecar for `/model.4/m.0/cv1/conv/Conv`.

The sidecar is exact:

- CPU0-3 correctness: pass
- mismatches: 0
- max_abs_diff: 0

The sidecar is not fast enough:

- direct sidecar: `56980.9 us`
- MMT4D 1-thread: `20544.9 us`
- MMT4D 4-thread: `5437.09 us`
- direct vs MMT4D 1-thread speed ratio: `0.360558x`
- direct vs MMT4D 4-thread speed ratio: `0.0954194x`

Primary reason:

`panel_build_mean_us=38901.3`, which dominates the direct sidecar.

Secondary node:

Not attempted because the primary node failed the minimum speed gate.

Next recommendation:

Do not integrate this Stage31 direct/sliding candidate. Prefer either:

1. a new proof lane that eliminates panel-build duplication before retrying direct Conv, or
2. return to current MMT4D/threaded selected-cut work and only revisit vmadot123 if a low-overhead sliding data layout is designed first.
