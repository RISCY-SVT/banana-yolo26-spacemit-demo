# Internal-R&D Distribution

The `0.9.2-stage59-final-internal-rd` bundle is restricted to authorized
internal research and engineering handoff. It includes the exact source model:

```text
manual_e2e_rep_conv_matmul_qdq.onnx
SHA-256 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
```

This inclusion follows direct user authorization for internal R&D only. It does
not establish ownership, clear third-party obligations, or authorize external
redistribution. Preserve `INTERNAL_R&D_ONLY.md`, `MODEL_PROVENANCE.md`,
`MODEL_LICENSE_RECORD.md`, and `SOURCE_MODEL_SHA256` with the file.

Use the runtime bundle when the source ONNX is not required. Both bundles
contain the same immutable prepared `package/` consumed by the executor.
