# ORT 2.0.6 identity reuse

Stage64 reuses the accepted Stage63 official archive read-only:

```text
asset: spacemit-ort.riscv64.2.0.6.tar.gz
bytes: 15002263
SHA-256: bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6
ORT version: 1.24.2+spacemit.a1
embedded ORT commit: 9bb02204b
SpacemiT EP package version: 2.0.6
```

Key library identities:

```text
libonnxruntime.so.1.24.2+spacemit.a1:
  93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3

libspacemit_ep.so.2.0.6:
  dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae
```

The archive, extracted tree, headers, and libraries match Stage63. No vendor
runtime file was replaced or patched. The accepted Stage63 plugin matrix is
therefore imported, with a fresh official-sample load smoke and ten exact
independent-plugin dispatches as the bounded non-regression check.
