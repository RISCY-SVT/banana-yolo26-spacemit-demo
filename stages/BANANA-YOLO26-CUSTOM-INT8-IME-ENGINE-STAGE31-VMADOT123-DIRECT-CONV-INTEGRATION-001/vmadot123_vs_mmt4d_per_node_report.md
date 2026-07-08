# vmadot123 vs MMT4D Per-Node Report

Node:

`/model.4/m.0/cv1/conv/Conv`

Shape:

`80x80x32 -> 80x80x16`, kernel `3x3`, stride `1`, pad `1`

Protocol:

- Board: BPI-F3/K1X
- Affinity: `taskset -c 0-3`
- warmup: 10
- runs: 100
- repeats: 5

Correctness:

`mismatches=0`, `max_abs_diff=0`, checksum stable.

Timing:

| Candidate | mean_us | stddev_us | CV% | Notes |
| --- | ---: | ---: | ---: | --- |
| Direct/sliding vmadot1/2/3 sidecar | 56980.9 | 11.5521 | 0.0202736 | single-thread sidecar |
| Current MMT4D 1-thread | 20544.9 | 82.4099 | n/a | same node |
| Current MMT4D 4-thread | 5437.09 | 29.5962 | n/a | current best |

Direct sidecar component timing:

| Component | mean_us |
| --- | ---: |
| panel_build | 38901.3 |
| kernel_compute | 15795.9 |
| correction | 201.322 |
| writeback | 1275.35 |

Speed gates:

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| same-thread direct vs MMT4D | >= 1.20x faster | 0.360558x | fail |
| direct vs current best threaded MMT4D | >= 1.15x faster | 0.0954194x | fail |

Conclusion:

The real-node direct/sliding sidecar is exact but not competitive. The main blocker is panel-build and duplicate-row scheduling overhead, not int32 correction.
