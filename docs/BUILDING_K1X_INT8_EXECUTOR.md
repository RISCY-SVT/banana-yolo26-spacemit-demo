# Building The K1X INT8 Executor

## Toolchain

The tested cross compiler is SpacemiT GCC 14.3.0. The selected executor flags
are:

```text
-march=rv64gcv_zvfh
-mabi=lp64d
-mtune=spacemit-x60
-funroll-loops
-O3
-DNDEBUG
```

The IME implementation is compiled only for the approved K1X objects and runs
only on CPU0-3. Do not broaden `-march` or add `-mcpu=spacemit-x60` without a
new exactness, disassembly, and full-model timing gate.

## Cross Build

Use the source-controlled Stage59 build wrapper. It configures the frozen
executor, camera demo, static library, and shared library in one build:

```bash
scripts/build_cross.sh \
  --build-root /data/build/banana-yolo26-k1x-demo-0.9.3 \
  --install-root /data/install/banana-yolo26-k1x-demo-0.9.3
```

The official configure sets `Y26_DEMO_OFFICIAL_K1X_RELEASE=ON` and
`Y26_K1X_ENABLE_IME=ON`. Configuration fails closed on a non-RISC-V target or
without IME, and the wrapper verifies the embedded IME/RVV/frozen-profile
capability marker. Installed artifacts are
`liby26_k1x_int8_executor.a` and `liby26_k1x_int8_executor.so`.
The install also contains `yolo26_k1x_int8` and the C11 ABI lifecycle probe
`y26_k1x_healthcheck`, plus `y26_k1x_demo`.

The release uses origin-relative runtime paths and deterministic archive
construction. Independent builds
under differently named build roots must produce byte-identical installed
libraries, executables, and headers. Compare complete install-tree SHA-256
inventories before publishing a handoff bundle.

## Host Tests

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -S custom_int8_engine \
  -B .deps/custom_int8_engine/build-host-release \
  -GNinja -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DY26_K1X_ENABLE_IME=OFF \
  -DY26_K1X_OFFICIAL_RELEASE=OFF \
  -DY26_K1X_BUILD_RESEARCH=ON
cmake --build .deps/custom_int8_engine/build-host-release -j8
ctest --test-dir .deps/custom_int8_engine/build-host-release --output-on-failure
```

The full package generator requires the prepared ONNX/NumPy environment. Its
accepted model SHA-256 is
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.
