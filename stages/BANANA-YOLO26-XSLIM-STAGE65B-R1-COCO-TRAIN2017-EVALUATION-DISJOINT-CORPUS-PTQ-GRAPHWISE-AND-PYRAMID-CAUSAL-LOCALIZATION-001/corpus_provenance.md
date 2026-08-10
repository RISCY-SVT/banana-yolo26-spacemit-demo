# Frozen calibration corpus

The corpus is the official COCO 2017 train split, acquired from the official
archive and retained for internal R&D only. The annotation archive SHA-256 is
113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268;
the train image archive SHA-256 is
69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929.

The parsed detection surface contains 118,287 images, 860,001 annotations,
80 categories, and eight license records. A deterministic qualified pool of
117,266 images was ranked from the annotation-archive hash and image ID.
Only the 3,014-image reserve/selection union was extracted. One internal
duplicate was excluded, leaving 3,013 unique decoded images.

All selected rows retain the COCO image ID, source URLs, license ID resolved
through the archive's own licenses array, category/object coverage, exact
file hash, and canonical pixel hash. No image bytes are tracked in Git,
exported in the result packet, synchronized to Drive, or authorized for
redistribution. No commercial-clearance claim is made.
