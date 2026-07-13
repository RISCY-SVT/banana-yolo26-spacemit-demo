# Stage 50 final-commit regression

The exact `ea993fb4255f12592380b975bd3cc6dbc73bea57` source reproduced the accepted contract and removed the prior diagnostic
binary provenance caveat. F0-F7 are exact at all 32 integer boundaries, FRM is restored, no
CPU4-7 IME executes, and internal conversions/float materializations remain zero.

- model5: 4826.757964 us mean, 4918.032450 us p95.
- model4-final to model8: 27261.068684 us mean, 27396.073300 us p95.
- package manifest: `2dbdbd18abe1ba126f12246b82c25821b9f74eb0ee9c324cb30aaaa062f64527`.

All predeclared regression ceilings passed.
