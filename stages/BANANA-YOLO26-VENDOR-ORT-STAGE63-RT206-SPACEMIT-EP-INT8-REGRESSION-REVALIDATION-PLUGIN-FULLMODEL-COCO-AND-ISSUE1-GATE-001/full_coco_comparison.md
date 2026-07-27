# Full COCO comparison

All Stage63 metric rows were evaluated with the accepted COCO val2017
annotations and the exact image IDs in each timing TSV. Historical rows
are retained only when their archive, model, evaluator, and fixed-subset
identity gates passed.

| Surface | Images | mAP50-95 | mAP50 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| rt206_cpu_fp32_full | 5000 | 0.404730651123 | 0.571261907189 | 0.197788572585 | 0.441452359314 | 0.586958794540 |
| rt206_cpu_int8_full | 5000 | 0.374594101159 | 0.539387780039 | 0.190490693707 | 0.415870903737 | 0.544466249019 |
| rt206_spacemit_fp16_full | 5000 | 0.404686635006 | 0.571409774907 | 0.197820280412 | 0.441256207899 | 0.586883081457 |
| rt206_spacemit_fp32_full | 5000 | 0.404745220420 | 0.571234716677 | 0.197822649667 | 0.441534564897 | 0.587006096386 |

The SpacemiT INT8 surface is absent because session creation aborts;
it is not represented as a zero-accuracy run. Cross-runtime final
tensor byte identity is not used as the correctness oracle.

The 2.0.6 CPU INT8 run matches the accepted 2.0.5 CPU control for all
5,000 per-image output and detection hashes and retains prediction SHA-256
`1dbc118383009fa68ddb1b786af68f56b38c473d8c03a23a782a38e4727d0b48`.
The changed 2.0.6 EP differs from the 2.0.4 EP at tensor level on every FP32
image and on 4,996 FP16 images. Aggregate COCO accuracy remains effectively
equal, but that does not make the runtime outputs byte-identical.

The annotation file SHA-256 is
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`;
evaluation used pycocotools 2.0.11.
