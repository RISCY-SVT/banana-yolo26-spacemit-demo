# Targeted policy proposals

No model is generated in Stage65B-R3. At most two later policy classes are justified.

## Policy A: all-S8 local robust-range policy

Apply an XSlim-supported local observer/range-correction policy to the two task-causal regions while retaining signed S8 QDQ throughout: the R0 stem/model.0-model.2 path and the R7 model.23 per-scale head prefixes, with priority on P4/P5 confidence activations. Candidate observer classes are bounded local percentile, MSE/KL, or bias/range correction; Stage R3 does not choose one without a generation/evaluation gate. Preserve six outputs, separate bbox/confidence branches, zero QLinear, zero UINT8 zero points, and explicit Conv kernel_shape.

## Policy B: bounded higher-precision/exclusion policy

If Policy A cannot recover the residual, leave only the proven R7 confidence-prefix region at higher precision or exclude it from quantization. This is secondary because it risks splitting a SpacemiT fused region and violates the resident custom engine's all-INT8 dataflow assumption. It requires explicit conversion/fallback accounting and is not provider-compatible evidence by itself.
