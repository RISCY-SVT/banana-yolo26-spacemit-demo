# Prefix contract

The prefix begins at normalized RGB `images` and ends at
`/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`.

The selected RGB path writes input codes directly into resident NCHWc8 storage;
it does not materialize a full float input tensor for `run_rgb`. Dense Conv
uses shape-exact output tails, including N4/N8/N16 where present, and joins the
accepted model4-final resident region without an internal layout adapter.
