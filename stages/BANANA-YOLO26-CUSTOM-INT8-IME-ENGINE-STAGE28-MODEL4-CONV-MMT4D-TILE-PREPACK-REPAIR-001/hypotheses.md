# Stage28 Pre-registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

H1: T1 decomposition will split each selected Conv node into non-overlapping pack/im2col, MMT4D compute, correction, writeback/copy, and thread/barrier/copy overhead buckets.

H2: If MMT4D compute dominates and pack/im2col/correction/writeback are small, tile/prepack repair cannot provide a material local win. In that case Stage28 must stop with a structural-limit decision instead of forcing a tile candidate.

H3: If correction/writeback/copy or thread/barrier overhead is material, T2 correction fusion / reduced copy can reclaim a measurable part of the Conv bucket while preserving ONNX-cut byte exactness.

H4: If the largest material Conv sub-bucket is tile/prepack/layout related, exactly one bounded tile/prepack candidate may be implemented and accepted only if it improves the targeted Conv sub-bucket by >=1.2x and preserves ONNX-cut correctness.

H5: `vmadot1/2/3` is not implemented in Stage28. It may be recommended only as a future proof lane if Stage28 proves structural MMT4D limitation, and the recommendation must be gated on Track B YOLO26 mAP/value evidence.
