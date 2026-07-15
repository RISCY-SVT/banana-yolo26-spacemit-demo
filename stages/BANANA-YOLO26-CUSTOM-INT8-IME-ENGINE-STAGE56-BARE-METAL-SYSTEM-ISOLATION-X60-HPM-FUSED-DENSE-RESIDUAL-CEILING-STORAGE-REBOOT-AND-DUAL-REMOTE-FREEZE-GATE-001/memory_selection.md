# Memory selection

Retain M0. Anonymous mmap, mlock/prefault, targeted MADV_HUGEPAGE, and MADV_COLLAPSE were functional; M3 obtained 8 MiB AnonHugePages, but M1-M3 did not clear the 0.5% mean or 10% p99.9 gate.
