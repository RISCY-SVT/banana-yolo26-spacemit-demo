# Attention V4 contract

The selected candidate writes exact Q48 Softmax results directly in the packed order consumed by the second IME MatMul, eliminating the intervening repack. It preserves exact normalization, tie behavior, and CPU0-3-only IME.
