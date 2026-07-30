# ReduceMax source comparison

Official XSlim 2.1.1 implements `ReduceMax_forward` by destructuring exactly
one entry from `values` and reading axes from attributes. Its tree already has
an adjacent `_get_reduce_inputs` helper, but ReduceMax does not call it.

At vendor commit `9a33f2f`, `ReduceMax_forward` calls:

```text
_get_reduce_inputs(op, values, min_input_opset=18)
```

The same helper is shared by the other reduction routes. It handles the
opset-18 axes input and preserves `keepdims` and
`noop_with_empty_axes` handling.

Measured consequence:

- official 2.1.1 fails all three tiny regression graphs after its opset
  conversion produces a two-input ReduceMax;
- the vendor reference exports and runs all three;
- the two pure ReduceMax controls are exact;
- the Conv + ReduceMax PTQ control has maximum absolute error
  `0.014914602041244507` against its floating-point host oracle.

The six-output YOLO split avoids executing ReduceMax inside XSlim, so the
official split lane is valid evidence even though direct E2E remains broken.
