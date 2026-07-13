# Attention contract

The two attention blocks contain four static-shape integer MatMul operations
and two Softmax/transpose operations. MatMul uses signed-storage correction,
packed right-hand tiles, exact Q62 requantization, and IME only on CPU0-3.
Softmax uses package-defined Q48 exponent tables, a fixed reciprocal, exact
integer division with round-to-nearest-even, and a frozen transpose. There is
no floating-point Softmax in the measured executor.
