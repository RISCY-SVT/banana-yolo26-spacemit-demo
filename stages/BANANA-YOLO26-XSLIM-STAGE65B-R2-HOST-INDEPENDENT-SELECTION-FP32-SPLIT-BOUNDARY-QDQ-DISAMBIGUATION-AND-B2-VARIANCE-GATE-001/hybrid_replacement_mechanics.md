# Hybrid replacement mechanics

Accepted evaluator source:

```text
vendor_ort_validation/stage65b_r1_evaluate.py
SHA-256: 79ad059411bb153f3abcb8d4abd0f1e79e5e04b12863fc121dc227d2fe89bd65
```

The replacement table is defined at lines 31-41. `run_hybrid` is at lines
281-350. For each shared preprocessed input, it executes the FP32 inference
session at line 303 and the candidate inference session at line 305. Each arm
then starts from `list(q_boundaries)` at line 308 and replaces only the listed
indices from `fp_boundaries` at lines 309-310. Every arm invokes the same tail
session through `tail_run` at line 312.

Consequently, H1-H8 replace complete boundary tensors after two independent
inference executions. They do not bypass only the candidate's final output
Q/DQ pairs. H8 replaces all six candidate slots with FP32 inference outputs;
H0 replaces none.

Frozen identities:

```text
FP32 inference: 72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8
B2 inference:   40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853
common tail:    18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3
```
