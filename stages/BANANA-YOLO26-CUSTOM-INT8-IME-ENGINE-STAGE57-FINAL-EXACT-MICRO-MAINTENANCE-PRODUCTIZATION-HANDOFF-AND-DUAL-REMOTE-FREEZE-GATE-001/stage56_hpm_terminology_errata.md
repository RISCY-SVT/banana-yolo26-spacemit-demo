# Stage56 HPM Terminology Errata

Stage56 HPM values are event counts per cycle. The L1D read-miss value is not a same-run miss/access ratio, and a DTLB miss ratio is unknown because the access event returned zero.

The supported conclusion is backend/structural/dependency-or-latency dominated, not frontend/I-cache/branch dominated on the measured event surface. Backend stalls are not attributed solely to Q62.
