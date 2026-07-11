# Island ROI Report

Measured same-session surfaces:

- model4 final activation/requant only: `2589.904 us`;
- model5 custom Conv plus exact postactivation: `26579.802 us`;
- equivalent isolated board ORT model5: `17862.861 us`;
- custom model5 delta: `+8716.941 us`, `48.799%` slower.

The model5 internal compute comparison is negative. Accordingly:

- model5 compute: negative;
- model4-to-model5 island internal: not benchmarked after gate short-circuit;
- island with entry/exit adapters: not benchmarked after gate short-circuit;
- paired Stage42/Stage43 full hybrid scaffold: not run after gate short-circuit.

This is not a claim about model FPS. It is a bounded rejection of this exact model5 implementation route.
