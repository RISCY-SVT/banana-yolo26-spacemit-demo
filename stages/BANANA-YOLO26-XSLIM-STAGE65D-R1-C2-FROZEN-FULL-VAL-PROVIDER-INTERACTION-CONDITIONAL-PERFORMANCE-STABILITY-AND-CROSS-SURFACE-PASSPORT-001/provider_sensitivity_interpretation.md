# Provider score/rank sensitivity interpretation

This analysis uses frozen full-val prediction JSON only. Spatial candidates are paired by deterministic mutual-best bbox IoU >= 0.5; it does not expose provider-internal rounding.

- C2 EP/CPU Top-100 membership Jaccard is `0.489375951` with `41292` crossings; B2 EP/CPU is `0.484204424` with `42216` crossings. C2 is not uniquely worse on this control-normalized surface.
- Top-300 Jaccard is `0.488150439` for C2 and `0.483102473` for B2. The recorded Top-300 crossing count is zero by construction because the serialized detector output is already capped at 300; it is not evidence of membership equality.
- C2 EP/CPU matched score absolute mean delta is `0.007369895486` with `11635` class changes; B2 is `0.006139322236` with `12280` class changes.
- EP minus CPU prediction-count deltas are `-4916` (C2) versus `600` (B2) at score 0.001 and `669` versus `209` at score 0.05. Counts are descriptive and are not an accuracy oracle.
- Population-level difference-in-differences classifies `10` task metrics provider-neutral, `2` inconclusive, and `0` material. Therefore the deterministic ranking differences do not establish a C2-specific material provider interaction.
- The data are compatible with confidence/rank sensitivity in both frozen models, but cannot prove an EP bug, exact rounding mode, or an LSB-level root cause.
