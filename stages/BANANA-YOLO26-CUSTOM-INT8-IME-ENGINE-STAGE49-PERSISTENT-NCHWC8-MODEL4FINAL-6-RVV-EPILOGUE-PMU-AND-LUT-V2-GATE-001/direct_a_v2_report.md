# Direct-A v2 report

P3 uses both fields of one `vlseg2e64` load for kernel-x 0/1 and a bounded `vlse64` load for kernel-x 2. It requires no full im2col and retains exact border/tail paths. P3 was the fastest integrated scout at 4824.669833 us. P4's shifted reuse was slightly slower. The gain is integrated, not inferred from a pack-only timer.
