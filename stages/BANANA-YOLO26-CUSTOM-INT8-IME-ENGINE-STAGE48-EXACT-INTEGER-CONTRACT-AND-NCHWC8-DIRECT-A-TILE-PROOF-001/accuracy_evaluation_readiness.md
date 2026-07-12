# Accuracy evaluation readiness

Full-model `K1X_INT8_V1` accuracy is not runnable in Stage48 because no complete
independent integer graph/export exists. Legacy float-QDQ COCO results remain
diagnostic and are not transferred to the new integer contract.

A later full-model gate must freeze the contract/package hash, preprocessing,
postprocessing, COCO manifest, and compare one complete independent integer
surface against the accepted FP32 and semantic INT8 references. No Stage48 COCO
or production-accuracy claim is made.
