# Calibration corpus provenance

## Decision

`stage65b-blocked-independent-calibration-corpus-missing`

No network dataset download was attempted. The existing dataset inventory
contains COCO val2017 and a 2,015-image calibration tree previously used by
Stage64. SHA-256 content matching proves all 2,015 calibration images are in
val2017, so that tree cannot establish independent calibration accuracy.

No COCO train2017 image tree or image archive is present. The only related
COCO train/val archive contains annotations, not training images.

The small YOLO26 project fixture collections contain 109 file rows but only
45 unique contents. Of those, 32 unique contents are val2017 images and 13
are synthetic, transformed, Ultralytics sample, camera, or private canonical
fixtures. They are not a documented licensed detection corpus and cannot
satisfy even C50, much less C1000.

Other inspected image roots contain at most 15 unique technical figures or
sample assets, not a qualifying detection corpus. Because corpus provenance,
minimum count, and zero-overlap requirements cannot all pass, selection and
preprocessing qualification were not performed.

Raw manifests and overlap joins remain under the task-local Stage65B
`calibration/` evidence directory.
