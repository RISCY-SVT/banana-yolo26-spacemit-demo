# Release reproducibility report

The handoff bundle was generated at
`/data/releases/banana-yolo26-k1x-int8-executor` from the independently
reproduced Stage52 E2c2 install tree and deterministic full-graph package.

## Identities

- Source input identity: `staged-tree:b35e9e803660806ab5db6a6ccb20a5d3060b82ac`.
- Package manifest SHA-256: `d3b4cb794f1373aa712d77bab177a5f7da58530361c9af58c0caf5bbcd6dc75f`.
- Release manifest SHA-256: `aeb7712395efee7f0a49ddc280e4e4002122aebaa4ba3e644b2ea6615741ea44`.
- Release checksum-file SHA-256: `82b5a4144a67208dbcf7047b1c8d31f4bcf65c8232e8b12da246f3442f1a1959`.
- Payload: 1201 checksummed files, 29361374 bytes.

Two clean cross-build/install runs produced byte-identical inventories. Both
inventory files have SHA-256
`790bd76450cbdbda11fe7f9ba47230556434beb9438a79e6c08a831967f9481c`.
The selected CLI, static library, and shared library hashes are respectively:

```text
29fbe2fc6c746c1e8d91fbbe7a7de0b69a717034eb8a8d666d4ccee9b504b904
365c7e954c6c9ddb5bd7857a2d2892fbf10b9efe674f1f10cafc0dad3fd562ff
f10aef68f82f35c224481593eaa2bd1675e33fadd5178d6fad7dffe77f41722a
```

`sha256sum -c release_sha256.txt` passed before deployment and again under
the board deployment root. The release contains no symlinks. Its source-tree
identity deliberately excludes the later evidence-only final commit metadata;
the final project commit and remote identity are recorded separately.
