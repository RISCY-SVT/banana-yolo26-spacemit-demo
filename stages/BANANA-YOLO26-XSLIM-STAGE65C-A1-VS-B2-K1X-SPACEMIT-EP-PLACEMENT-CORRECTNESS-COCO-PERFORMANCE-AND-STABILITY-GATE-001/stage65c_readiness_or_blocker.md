# Stage65C readiness or blocker

Status: `blocked-at-h500`.

A1 preserves the B2 SpaceMIT partition shape and reproduces a positive H500
mAP50-95 delta on the EP. It nevertheless fails mandatory board-candidate
guards:

- A1 EP versus B2 EP AR-small delta is `-0.009086253892`, below `-0.005`.
- A1 EP versus B2 EP AR-large delta is `-0.017721888749`, below `-0.005`.
- A1 CPU versus EP mAP50-95 difference is `0.001378162000`, above `0.001`.
- A1 CPU versus EP AP-large difference is `0.008548174752`, above `0.003`.
- A1 CPU versus EP AR-small and AR-large differences also exceed `0.003`.

The Stage contract therefore closes before full val2017, matched ABBA timing,
and long soaks. A1 is not ready for runtime-baseline review. B2 remains the
frozen vendor-lane control; no default runtime or artifact is changed.
