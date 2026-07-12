
# Model5 direct-layout performance

Protocol: board CPU0-3, governor `performance` at 1.6 GHz, warmup 10, 100 runs,
5 repeats. Every arm produced the fixed expected output hash.

Selected R3 is M12xN16, spatial partition, four workers, scalar exact integer
epilogue, and `vlseg2e64` direct delivery. Mean is `6516.213018 us`,
median `6327.420000 us`, p95 `8359.414750 us`,
and `36.206551 GMAC/s`. Sparse scheduler excursions make the
per-sample CV `11.915021%`; the five repeat means have CV
`0.974733%`,
so the repeated central result is stable while all outliers remain in raw logs.

This is `4.049592x` faster than Stage47 R0 (`26388.005044 us`) and
`44.310631%` lower than the resource-matched B120 ORT model5
reference (`11701.000 us`). It therefore satisfies the predeclared
ORT-competitive threshold.

The NCHW-to-NCHWc8 conversion is excluded by contract and separately measured:
entry `30343.690198 us`, exit `7836.849598 us`. A later persistent-layout slice
must prove that these conversions are not paid per operator. No end-to-end or
production result is claimed.
