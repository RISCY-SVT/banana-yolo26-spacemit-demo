# Independence classification

COCO train2017 is **evaluation-disjoint** from COCO val2017 by image ID,
original JPEG SHA-256, and canonical decoded-pixel SHA-256.

It is **not training-independent**: the canonical model provenance proves that
COCO train2017 was the training corpus. Results therefore isolate val2017
calibration/evaluation overlap and PTQ factor effects, not independence from
the model's learned training distribution.

Open Images V7 would have been an external-domain control. Its metadata was
captured, but image acquisition was not possible through an allowlisted
official object surface, so no Open Images PTQ lane is represented as tested.
