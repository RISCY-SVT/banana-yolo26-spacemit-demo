# Source Model Not Redistributed

The source ONNX and upstream trained weights are intentionally absent from this
distribution. The repository does not contain a sufficiently closed provenance
and redistribution-license record for those trained weights. Stage58 therefore
does not infer redistribution permission from source-code availability.

The prepared `package/` directory is included as the runtime model explicitly
authorized for this engineering handoff. Its accepted source-model identity is:

```text
manual_e2e_rep_conv_matmul_qdq.onnx
SHA-256 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
```

To regenerate the package, an authorized maintainer must obtain the exact source
artifact through the project's controlled model provenance process, verify this
SHA-256, and run the source-controlled package tooling. Do not substitute another
internally consistent model or package.
