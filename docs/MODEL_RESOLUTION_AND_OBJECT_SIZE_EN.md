# Model Resolution And Object Size

## Fixed Tensor

The released executor accepts exactly one model tensor size: 640x640 RGB. A
camera can capture another supported resolution, but the demo aspect-preserving
resizes and letterboxes every frame to 640x640 with pad value 114.

## COCO Evidence

Stage58 matched accepted predictions to non-crowd COCO val2017 ground truth at
the same class, confidence at least 0.25, and IoU at least 0.50. This is a
diagnostic recall envelope, not COCO AP and not a product guarantee.

| Object group | Ground truth | Matched recall |
|---|---:|---:|
| COCO small, area below 32² | 12,766 | 0.1794 |
| COCO medium, 32² to 96² | 13,253 | 0.5082 |
| COCO large, at least 96² | 10,316 | 0.7991 |

Recall by the shorter object side after the exact 640 letterbox was 0.0302 for
0-8 px, 0.1141 for 8-12 px, 0.1882 for 12-16 px, 0.2458 for 16-24 px, 0.2988
for 24-32 px, 0.3921 for 32-48 px, 0.4872 for 48-64 px, 0.5717 for 64-96 px,
and 0.7685 for at least 96 px. Every bin and sample count is in
`resolution_coco_bins.tsv`.

The first measured shorter-side bin above 50% recall is 64-96 px. This is only
a dataset-level observation at the stated threshold and matching rule. It is
not a universal minimum reliable object size.

## Real Camera Caveat

There is no universal minimum number of pixels for all objects. Reliability
depends on class, contrast, motion blur, occlusion, lens, distance, confidence
threshold, and the pixels occupied after 640x640 letterboxing. Stage58's fixed
wall-poster view produced repeated `person`, `suitcase`, and `umbrella`
detections. The smallest accepted shorter-side observation was 38.184 pixels,
inside the 32-48 pixel bin. These are detector observations without independently
staged ground truth, distance, or scene variation, so they do not establish a
reliability threshold. The requested three-scene/five-class study could not be
performed remotely; `resolution_camera_observations.tsv` preserves the bounded
evidence instead of inventing scenes.
