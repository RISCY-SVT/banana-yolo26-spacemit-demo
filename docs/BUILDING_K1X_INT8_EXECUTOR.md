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

Use the source-controlled release build wrapper. It passes the compiler
contract explicitly to both static and shared CMake trees:

```bash
scripts/k1x-int8-executor/build.sh \
  "$PWD" \
  "$PWD/.deps/custom_int8_engine/release-build" \
  "$PWD/.deps/custom_int8_engine/release-build/install"
```

The wrapper builds both `BUILD_SHARED_LIBS=OFF` and `ON`. The target name is
`y26_k1x_custom_int8_engine`; installed artifacts are
`liby26_k1x_int8_executor.a` and `liby26_k1x_int8_executor.so`.

## Host Tests

```bash
cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host-release \
  -GNinja -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DY26_K1X_ENABLE_IME=OFF -DY26_K1X_ENABLE_TESTS=ON
cmake --build .deps/custom_int8_engine/build-host-release -j8
ctest --test-dir .deps/custom_int8_engine/build-host-release --output-on-failure
```

The full package generator requires the prepared ONNX/NumPy environment. Its
accepted model SHA-256 is
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.
