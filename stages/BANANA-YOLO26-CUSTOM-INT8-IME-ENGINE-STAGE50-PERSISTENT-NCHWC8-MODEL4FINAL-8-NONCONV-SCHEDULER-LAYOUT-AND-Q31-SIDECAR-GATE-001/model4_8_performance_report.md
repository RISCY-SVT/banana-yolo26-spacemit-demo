# Model4-final to model8 performance

Under CPU0-3, 10 warmups, 100 runs, and 5 repeats, the resident custom slice measured **27174.621676 us mean** and **27269.385650 us p95**. The equivalent B120 ORT CPU cut measured **57909.967262 us mean** and **58031.158848 us p95**. Custom deltas are **-53.074362% mean** and **-53.009062% p95**, satisfying the strong-net-positive band.

The custom path has zero internal layout conversions and zero float materializations. Entry plus internal plus exit is a non-paired diagnostic sum of 62920.626352 us and remains slower than ORT. This does not contradict the resident-layout result: the architectural requirement is to retain the layout across a larger custom region rather than wrap this slice with adapters.
