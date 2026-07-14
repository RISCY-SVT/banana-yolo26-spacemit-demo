# Release update

The Stage53 optimized-research bundle is preserved. Stage54 qualifies for an updated bundle at `/data/releases/banana-yolo26-k1x-int8-executor/stage54-optimized-research` because mean improved 30.211342%, exactness and COCO identity pass, the 10000-run soak passes, and API/CLI compatibility is retained.

Release tree-manifest SHA-256: `e636c56fe4c65a2336928cea57c62c5b48930509fd73b61f229097b3a67e8749`. Checksum-file SHA-256: `fc069c7ae3032ea104e9cae9b6c0cd74a4583cce741266c204eb9f0450bea1bb`. Packaged CLI SHA-256: `873074863c1d051bbdd9695e15575db49a0aa930a2b4c2d7c51f55a2dbb11523`. Source commit: `233bd46fecbb6b4396e4d869253ddca9ba5dfc6f`. Package manifest: `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.

On-board checksum verification, loader resolution, C API smoke, CLI smoke, and bundled compatibility/low-latency benchmarks passed from the deployed NVMe release root.

The bundle distinguishes condition-variable compatibility from dedicated-board epoch-spin low latency. It is an optimized research handoff, not production-ready.
