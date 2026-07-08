# Next Stage Decision

decision: `STOP_CV2_PIPELINED_VMADDOT_FOR_NOW`

Reasons:

```text
- Existing accepted wrapper `smt.vmadot` path remains board-executable.
- Stage34 direct inline/register-blocked/pipelined shapes trap with SIGILL.
- No `/model.4/cv2` pipelined candidate reached correctness or timing.
- Current selected-cut baseline remains byte-exact and stable.
```

The next local repair lane should not be another raw `smt.vmadot` pipeline attempt. Recommended Stage35 focus:

```text
1. Re-attribute output_quantize_us (~7070 us / ~17.6%) and thread_overhead_us (~5243 us total).
2. Choose exactly one low-risk local repair:
   - output QuantizeLinear local repair if it is still material and exact;
   - or thread/barrier overhead reduction if overhead is proven locally reducible.
3. Preserve same-input ONNX-cut byte equality and FRM robustness.
```

No graph expansion is recommended until this local selected-cut bucket decision is closed.
