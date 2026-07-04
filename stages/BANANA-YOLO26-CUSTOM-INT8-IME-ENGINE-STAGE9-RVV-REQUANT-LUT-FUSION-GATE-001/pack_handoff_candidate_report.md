# Pack Handoff Candidate Report

classification: sidecar-pass-not-integrated

Candidate: `A5_fused_requant_lut_packA`

Implemented sidecar helpers:

- `y26_activation_packa_1x1_mmt4d_4x8_from_nhwc`
- `y26_activation_unpacka_1x1_mmt4d_4x8_to_nhwc`

Correctness:

- Small fixture pack/unpack: mismatches `0`.
- Full-shape Conv2 1x1 handoff pack/unpack in Stage 9 bench: mismatches `0`.

CPU0 sidecar timing:

| metric | value |
|---|---:|
| full-shape Conv2 1x1 packA handoff_us | 4059.95 |
| packA checksum | -99373052 |
| unpack mismatches | 0 |

Decision:

A5 was not integrated into the selected runner in Stage 9. A2 already reduced activation share below the 40% gate, so direct packA handoff can be deferred to a later integration/dataflow stage.
