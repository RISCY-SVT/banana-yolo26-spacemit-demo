# Attention execution proof

The selected attention MatMul invokes approved IME symbols on CPU0-3. The Stage54 V2 indexed-gather route assembled but trapped on board, so no unsupported vector route is selected and exact Stage53 MatMul/Softmax dataflow remains authoritative.
