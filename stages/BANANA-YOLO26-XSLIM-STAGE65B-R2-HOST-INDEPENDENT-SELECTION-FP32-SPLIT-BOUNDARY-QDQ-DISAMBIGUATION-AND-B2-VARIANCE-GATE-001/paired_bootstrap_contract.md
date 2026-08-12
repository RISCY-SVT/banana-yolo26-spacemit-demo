# Paired bootstrap contract

- Unit: image ID, sampled with replacement.
- Replicates: 1000; seed: 65002.
- Duplicate draws are represented by repeated exact COCO match records. A
  literal unique synthetic-image-ID remap was checked and agreed to <=1e-12.
- Every replicate reruns COCO accumulation; per-image pseudo-AP is not used.
- Interval: percentile 95%. Metrics: mAP50-95, mAP50, AP-small/medium/large.
- The vectorized precision-envelope implementation was proven array-identical
  to the original implementation on the complete H500 mandatory matrix.
