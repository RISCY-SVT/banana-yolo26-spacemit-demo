# Plugin and tail capability

Status: `structurally-possible-not-implemented`.

The shipped plugin contract exposes both full custom-operator registration (`AddOperator`) and dispatch overlay registration (`AddDispatch`).

The accepted CPU tail has `34` nodes, six FP32 inputs, and one `1x300x6` FP32 output. A CPU custom op or external C++/RVV stage is structurally conceivable. SpaceMIT ownership, exact numerical equivalence, fused placement, and speed are not proven. No source, custom op, or model was created.
