# Attention V3 contract

A3 preserves package Q48 exp values and exact denominator/normalization. It forms byte offsets under e16, uses a legal e64,m4 indexed load, and leaves both IME MatMuls on CPU0-3. No approximate reciprocal, polynomial, FP16, or arithmetic-contract change is used.
