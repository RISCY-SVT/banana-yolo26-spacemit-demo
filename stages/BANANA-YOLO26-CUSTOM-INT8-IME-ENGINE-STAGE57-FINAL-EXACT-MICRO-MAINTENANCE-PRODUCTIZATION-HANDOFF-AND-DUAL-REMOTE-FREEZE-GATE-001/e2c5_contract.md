# E2c5 Exact Contract

E2c5 consumes two neighboring raw C4 groups as independent vector chains. It widens int32 to int64, adds corrected bias, applies exact Q62 `vsmul` RNE, adds the output zero point, clamps/narrows, optionally gathers LUT bytes, and stores two contiguous C4 groups without `vslideup` or a corrected[8] stack array.

The governing arithmetic remains `K1X_INT8_V1`; ambient vector state is restored.
