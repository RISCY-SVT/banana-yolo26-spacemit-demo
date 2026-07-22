# Profile Guide

## Selection Policy

R640 is the only default and accepted exact release profile. Omitting
`--model-resolution` selects R640. The integrated research build exposes eight
additional Q0 profiles only through an explicit resolution and exact package
manifest; it never auto-selects or silently falls back.

| Resolution | Pure model mean | Pure FPS | mAP50-95 | Status |
|---:|---:|---:|---:|---|
| 640 | 131.155 ms | 7.624 | 0.370741 | accepted exact default |
| 512 | 94.117 ms | 10.625 | 0.347630 | experimental Q0 |
| 448 | 64.266 ms | 15.560 | 0.332627 | experimental Q0 |
| 416 | 55.808 ms | 17.919 | 0.317789 | experimental Q0 |
| 384 | 47.380 ms | 21.106 | 0.306537 | diagnostic; 6.42 AP-point loss |
| 352 | 40.797 ms | 24.511 | 0.289709 | experimental Q0 |
| 320 | 34.209 ms | 29.232 | 0.276269 | experimental Q0 |
| 256 | 24.350 ms | 41.067 | 0.231262 | diagnostic lower bound |
| 768 | 197.530 ms | 5.062 | 0.373550 | +0.281 AP point estimate; mixed size/class effect |

These are preprocessed pure-model measurements, not camera or sensor FPS.
R384 and R768 are not deployment-promoted.

Example explicit research selection:

```bash
y26_k1x_demo \
  --package /data/packages/r384 \
  --expected-manifest-sha256 a278db8b4f5aa3046ea8e65808e2978af88e4a2d115275829d6dab0720e33c8a \
  --model-resolution 384 \
  --source image:/data/input.jpg \
  --headless
```

`PROFILE_PROVENANCE.tsv` is the machine-readable identity authority.
