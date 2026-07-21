# ABI and Dynamic-Link Audit

The same SpacemiT 1.1.2 cross toolchain rebuilt the frozen 0.9.2 baseline and
the 0.9.3 candidate. Their dynamic symbol inventories are exactly equal: 15
public C functions plus the `Y26_K1X_ABI_1` version node. Every function remains
bound to `Y26_K1X_ABI_1`; no public symbol was added, removed, or renamed.

The candidate SONAME is `liby26_k1x_int8_executor.so.1`. Its five `DT_NEEDED`
entries are unchanged: `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`,
`libc.so.6`, and `ld-linux-riscv64-lp64d.so.1`. There is no `DT_RPATH`,
`DT_RUNPATH`, `DT_TEXTREL`, or unexpected text relocation.

The official capability marker is
`0.9.3/K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001/abi1/ime1/rvv1/frozen1`.
Raw `nm`, `readelf`, relocation, and binary-hash evidence is retained in the
Stage60M raw log root.

| Artifact | SHA-256 |
|---|---|
| shared library | `84490b92ba8c89b7d989250b3ea487aeb40320fcf4e0fac797f07fb6329ddd91` |
| static library | `06082b4f1c6be95629caa0e135d9a42bf8ca32660ea3f6692cb405996f0f620e` |
| release CLI | `a26d08af070b8a27a3f806df3ed554b041830ed034c4da7663a6d5f43074d431` |
| healthcheck | `4b82ff86aecbb07d1a5647bb7a92b9f9f3711a877ac5b54ed37fa30c75dfe2b4` |
| camera demo | `16bc79d6775048649ebeed58ae5347ae61f10883c1cf7efa02b78a7e1b23adb0` |
