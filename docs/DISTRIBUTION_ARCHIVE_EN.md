# Distribution Archive

Stage59 publishes separate runtime and internal-R&D deliveries, each as a
deterministic tar archive and ZIP:

```text
banana-yolo26-k1x-int8-executor-0.9.3-runtime-riscv64.tar.gz
banana-yolo26-k1x-int8-executor-0.9.3-runtime-riscv64.zip
banana-yolo26-k1x-int8-executor-0.9.3-internal-rd-riscv64.tar.gz
banana-yolo26-k1x-int8-executor-0.9.3-internal-rd-riscv64.zip
```

Extract under the board NVMe `/data`, then verify from the extracted root:

```bash
sha256sum -c SHA256SUMS
export LD_LIBRARY_PATH="$PWD/lib:$PWD/opencv/lib"
bin/y26_k1x_healthcheck --build-info
```

The archive contains the shared and static executor, ABI1 header, CMake and
pkg-config metadata, healthcheck and CLI, camera/image/video demo, COCO labels,
prepared immutable `package/`, known fixture, scripts, examples, documentation,
OpenCV 4.13 demo runtime closure, licenses, SBOM, and curated outputs.

The executor libraries do not depend on OpenCV. Only `y26_k1x_demo` uses the
bundled OpenCV component. The runtime bundle excludes the source ONNX. The
separately marked internal-R&D bundle includes the exact
`manual_e2e_rep_conv_matmul_qdq.onnx` under the direct internal-use
authorization, with provenance and license caveats. External ONNX
redistribution is not cleared. The prepared runtime package is included in both
bundles.

No archive path is absolute, no symlink escapes the root, and clean-extract
tests do not use repository build paths. Runtime assets, logs, screenshots, and
recordings belong under `/data`; eMMC is not selected.
