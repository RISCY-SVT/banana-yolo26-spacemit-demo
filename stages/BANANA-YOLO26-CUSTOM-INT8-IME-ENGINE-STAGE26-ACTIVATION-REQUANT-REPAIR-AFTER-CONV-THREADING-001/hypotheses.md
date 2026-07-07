# Pre-registered Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001

H1: The post-C1 activation/requant bucket is dominated by branch1 activation/requant and related add-slot scalar work. It is around 32.8 ms / 36.78% in the Stage25 selected cut.

H2: A per-boundary u8->f32 / u8->s8 SiLU LUT or equivalent exact LUT removes scalar std::exp from the hot path, remains byte-exact against the same-input ONNX cut, and reduces the activation/requant bucket by at least 3x if branch1 activation is the dominant sub-bucket.

H3: The selected activation/requant repair must be RNE/frm robust in the real runner path and must not rely on ambient frm.

H4: If branch1 activation is not the dominant sub-bucket after Stage26 replay, choose exactly one local lane based on measured buckets, or stop with a decision report. Do not optimize by intuition.

H5: Any accepted repair must run through the real runner API and preserve same-input ONNX-cut byte equality, not only internal scalar equality.
