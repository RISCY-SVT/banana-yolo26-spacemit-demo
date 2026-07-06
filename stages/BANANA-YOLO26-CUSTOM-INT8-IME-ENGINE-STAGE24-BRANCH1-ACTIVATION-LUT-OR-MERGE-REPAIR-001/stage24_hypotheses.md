# stage24_hypotheses

H0: Stage23 runner API + ONNX cut replay remains bit-exact with mismatches=0, max_abs_diff=0, SHA/checksum stable.

H1: Non-overlapping bucket attribution remains >=99% and identifies one dominant local repair lane.

H2: If merge/post-Concat QDQ/dataflow remains >=30% of total, a local exact RVV/fused merge repair can reduce merge_total_us by >=1.5x without changing ONNX-cut output bytes.

H3: If activation/requant exceeds merge and is >=30%, branch1 activation/requant LUT/RVV repair is the selected lane instead.

H4: If Conv exceeds 45-50% while merge/activation are below local repair thresholds, Stage24 must stop after decision and recommend Conv/threading/tile work, not implement it here.

H5: Any repair must live in the real runner API path, not only in the benchmark wrapper.

H6: RNE/frm robustness must remain valid for the accepted runner path under ambient RNE/RTZ/RDN/RUP/RMM.
