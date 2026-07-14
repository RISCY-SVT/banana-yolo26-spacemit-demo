# Stage53 final report

Technical classification: `stage53-structural-ceiling-strong-positive`.

Contract `K1X_INT8_V1`, profile `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`, model SHA-256 `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.

The selected route measured 239884.016000 us mean and 242452.000000 us p95. It is 52.369614% faster than the reproduced Stage52 full executor.

Matched B120 ORT measured 456266.315376 us mean; the selected custom route is 47.424562% faster.

COCO prediction SHA-256 is `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda` and package manifest SHA-256 is `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.

The byte-identical 5,000-image prediction surface preserves mAP50-95 0.3707408944391919 and the accepted preferred accuracy classification.

The selected 10,000-run soak recorded p99 250091.230000 us, p99.9 260405.010000 us, and maximum 263494.000000 us.

The calibrated cost-model error is +0.168061%.

Raw command, build, correctness, timing, COCO, and publication evidence is rooted at `/data/ncnn-logs/ai-team/2026-07-14/2026-07-14_08-08-50__codex__BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE53-FULL-EXECUTOR-STRUCTURAL-HOTSPOT-REPAIR-DUAL-REMOTE-AND-COST-MODEL-GATE-001__stage53-structural-hotspot-repair`.

No production, 20 FPS, training, student-selection, Q31, RT205, or co-design claim is made.
