# ORT Runtime And ABI Report

The compile-time API, linked ABI, provider inventory, and execution evidence are distinct. Details are tabulated in `ort_runtime_matrix.tsv`.

## Host

- runtime: ORT `1.27.0`, build commit `8f0278c77b`, Release.
- library SHA-256: `b5c9d4f124d24707f514dad926dc181820807178855df1c528e3addb2dd0e6f7`.
- header: vendor API 21 header, SHA-256 `591bb85c454386ddb2eca164cee1d60c94f9548f8d082c965ed0c7715a311b30`.
- runner requires symbol namespace `VERS_1.27.0`.
- registered providers: Azure and CPU. Neither was explicitly appended; CPU is the default session provider.
- final runner SHA-256: `04f8b6f0e80622fbc0eb780b536fa86dc42a45365ffae30a74778456381eb78e`.

## Board

- runtime: ORT `1.20.2+spacemit`; build-info API returns only `ORT_BUILD_INFO`.
- library SHA-256: `5a28c8128a7b1ed9cb29357f42eb7a2a45eb1b23d8791c2fee1eaf0151546238`.
- runner requires symbol namespace `VERS_1.20.2` and loads the intended library under controlled `LD_LIBRARY_PATH`.
- registered provider: CPU only.
- final runner SHA-256: `e8d53f2e32ae789db2c88fc3ac7f02048dc55351e7e197bf4012a52bfa3a9e8b`.
- compiler: SpacemiT GCC `14.3.0`, `-march=rv64gcv_zvfh -mabi=lp64d`.

The header API accepted all Stage42 session calls used by the runner: graph optimization, sequential/parallel execution mode, thread counts, profiling, logging, memory-pattern policy, CPU arena policy, and spinning configuration. API availability and successful linking do not imply matching kernels or arithmetic.

## Deployment caveat

The cross binary retains an absolute build-tree RPATH. The board loader result is controlled and proven, but this RPATH is not accepted as a deployable runtime contract. A future packaging stage must use a controlled `$ORIGIN` layout or strip/replace the build path.

Raw `readelf`, version-info, `ldd`, hashes, and CMake evidence are in command logs `0040`, `0041`, and `0112`.
