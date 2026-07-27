# 2.0.5 to 2.0.6 binary change

The package inventory contains 61 byte-identical paths, six changed paths,
24 additions, and two removals.

## Core

`libonnxruntime.so.1.24.2+spacemit.a1` is byte-identical:

```text
93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3
```

Its exports, undefined symbols, SONAME, dependencies, and embedded ORT build
commit `9bb02204b` are unchanged.

## Provider

`libspacemit_ep.so.2` changed from 5,703,872 to 5,962,176 bytes:

```text
2.0.5  3927b51f79f8d2142ff98708183aa9b24b47d6941533499035193a630042a41d
2.0.6  dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae
```

The provider adds 169 defined symbols and removes seven. Most relevant to
issue #1, previously unresolved public plugin methods are now defined.

## Package

The package version/date headers change from `2.0.5` / `2026-07-03` to
`2.0.6` / `2026-07-24`. The wheel is rebuilt and renamed. A complete top-level
plugin sample tree and scripts are added. Existing general ORT headers,
binaries, and most samples are unchanged.

Classification: **binary/package update built from an identical public source
tag**. Closed-binary source provenance is unknown.
