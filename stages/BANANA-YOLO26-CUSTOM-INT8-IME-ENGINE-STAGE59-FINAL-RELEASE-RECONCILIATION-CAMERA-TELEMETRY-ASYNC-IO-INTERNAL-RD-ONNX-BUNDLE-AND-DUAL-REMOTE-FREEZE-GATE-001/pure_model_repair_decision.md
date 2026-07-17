# Pure-Model Repair Decision

**Selected:** make the accepted compiler contract intrinsic to the official
release object target and require the fail-closed official K1X CMake option.

**Rejected:** arithmetic changes, model/package changes, new kernel routes, or
benchmark-specific environment compensation.

The controlled clean Stage58 rebuild measured 133834.854 us mean and
135434.950 us p95, satisfying the preferred Stage59 thresholds of 135500 us and
137000 us. A final 0.9.2 rebuild and A/B is required after all maintenance
source is frozen; its result remains the final selection authority.
