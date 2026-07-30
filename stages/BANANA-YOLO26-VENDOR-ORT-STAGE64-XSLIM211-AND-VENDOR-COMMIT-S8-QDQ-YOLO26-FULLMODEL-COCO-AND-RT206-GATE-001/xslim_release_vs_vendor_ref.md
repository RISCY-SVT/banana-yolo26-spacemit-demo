# XSlim 2.1.1 versus vendor reference

## Immutable refs

| Lane | Commit | Tree | Declared version |
|---|---|---|---|
| official 2.1.1 | `c246694a1eba8d7689c43ba7b5f469bb0cb29c95` | `0ff33a755ffe47852fac19ed349ccfc8dc580498` | 2.1.1 |
| vendor reference | `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` | `05d2c8425ab8587abf401fa5976a08d008fdd719` | 2.1.2 |

The official PyPI wheel is 312,873 bytes with SHA-256
`e01cd8b3c7070c038ed60415b30dfe1e35140de0de6725208b2eaa0f871069b3`.
The sdist is 322,572 bytes with SHA-256
`9804d5c473b9e79f391a645c403fd50dc68f0334ff07b4408692c4359f4f235c`.

The vendor-reference wheel was built from the exact commit in an isolated
worktree. Its SHA-256 is
`eb78f2f1cf98e94b3e214397aaa0bef16fe2ad53d318fe032f990e2f38d6d488`.

## Source delta

The exact diff changes 17 files with 2,074 insertions and 302 deletions.
Relevant areas are:

- generalized reduction input handling, including ReduceMax;
- QKV, split/slice, layer-normalization, and window graph passes;
- ONNX parser and simplification behavior;
- FP16 conversion and pipeline handling;
- expanded operator/pass tests;
- version metadata from 2.1.1 to 2.1.2.

The quantizer policy file itself is not changed by this commit. Stage64 still
audits emitted ONNX dtype/zero-point/granularity rather than assuming identical
models from that source fact.

The complete source patch is stored as
`xslim_release_vs_vendor_ref.patch`.
