# Model10 through model22 implementation

The executor implements the complete frozen model10-through-model22 region. It includes dense resident Conv, eight grouped/depthwise Conv nodes, two attention blocks, four static-shape MatMul nodes, two fixed-point Softmax surfaces, the exact nearest-neighbor Resize branches, Add/LUT/Concat transforms, and all required views.

Attention MatMul uses packed signed INT8 operands and IME on CPU0-3 when optimized mode is selected. Softmax uses the package-defined monotonic Q48 exponent table, exact unsigned division with round-to-nearest-even, and the frozen transpose. Grouped Conv retains an exact four-worker NCHWc8 direct fallback; it does not masquerade as a selected vector depthwise kernel. Resize implements only the ONNX mode present in the accepted graph.
