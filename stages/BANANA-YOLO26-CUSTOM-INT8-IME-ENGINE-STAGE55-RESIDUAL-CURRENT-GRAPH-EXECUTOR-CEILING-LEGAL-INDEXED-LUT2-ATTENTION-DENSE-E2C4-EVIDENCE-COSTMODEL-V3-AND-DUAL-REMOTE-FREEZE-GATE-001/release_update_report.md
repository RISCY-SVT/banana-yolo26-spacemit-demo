# Release update

Status: `stage55-optimized-research-bundle-created`. Stage54 remains preserved as historical optimized research. Stage55 is eligible because mean improved by at least 5%, exactness/COCO passed, and the 10000-run soak passed. Compatibility and low-latency profiles remain distinct. Neither is labeled production-ready.

Release manifest hash: `a884985eb0946e7c74793bc434a1f8b22e3926b978c1d6c3ba9231112854daa4`. Checksum-file hash: `5f0b01288c43191584e376103219b86bebb5c6d06c948d6107eda4d38be00b98`.

Independent checksum verification passed from the release root. The bundle was deployed under board NVMe `/data/k1x-yolo26-int8-executor/stage55-optimized-research`; the C API and CLI compatibility/low-latency smoke runs produced the accepted exact output.
