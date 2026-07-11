# Stage46 prompt: K1X student architecture and training-preparation gate

Stage ID: `BANANA-YOLO26-K1X-STUDENT-416-512-ARCHITECTURE-SPEC-AND-TRAINING-PREPARATION-001`

Start from the final Stage45 commit. Freeze the Stage45 model, 500-image accuracy
surface, measured K1X operator LUT, and fixed preprocessing. Without running a full
training campaign, define and validate two exportable student blueprints:

1. 416 latency-primary, 0.65-0.90 GMAC envelope.
2. 512 accuracy-fallback, 1.0-1.35 GMAC envelope.

Generate graph-level latency estimates from measured K1X primitives, tile-align
channels, select two- versus three-scale head by measured LUT, specify teacher,
distillation/QAT/pruning recipe, and produce a tiny untrained export/scheduler
contract only where useful. Do not claim accuracy or 20 FPS. Authorize training
only in a separate human-approved stage after architecture, dataset, compute,
accuracy, and acceptance contracts are reviewed.
