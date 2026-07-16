# Distribution Archive

The Stage58 delivery is available as both a deterministic tar archive and ZIP:

```text
banana-yolo26-k1x-int8-executor-0.9.1-riscv64.tar.gz
banana-yolo26-k1x-int8-executor-0.9.1-riscv64.zip
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
bundled OpenCV component. The source ONNX is intentionally not redistributed:
the trained-weight redistribution provenance was not sufficiently closed for
this handoff. `model/MODEL_SOURCE_NOT_REDISTRIBUTED.md` records its SHA-256 and
regeneration identity. The prepared runtime package is included unconditionally.

No archive path is absolute, no symlink escapes the root, and clean-extract
tests do not use repository build paths. Runtime assets, logs, screenshots, and
recordings belong under `/data`; eMMC is not selected.
