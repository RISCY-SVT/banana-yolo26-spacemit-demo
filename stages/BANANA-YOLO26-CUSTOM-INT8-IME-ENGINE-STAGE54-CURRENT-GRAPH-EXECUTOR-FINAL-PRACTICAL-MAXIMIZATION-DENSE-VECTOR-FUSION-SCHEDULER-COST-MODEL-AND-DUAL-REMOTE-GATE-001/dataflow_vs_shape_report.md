# Dataflow versus shape

The mandatory same-shape comparison does not support a single resident/non-resident explanation. High-M low-K 1x1 rows are mixed A-delivery/epilogue limited; N80 rows add tail and store pressure; stride-1 3x3 is mainly arithmetic/epilogue limited; stride-2 benefits from P3 segmented direct delivery.

The selected dispatcher therefore specializes exact shape classes instead of extending one resident facade indiscriminately.
