# Host Scalar Correctness Report

## Native build

Command family:

```bash
env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host-native \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++

env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  cmake --build .deps/custom_int8_engine/build-host-native -j"$(nproc)"

env -u CC -u CXX LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  ctest --test-dir .deps/custom_int8_engine/build-host-native --output-on-failure
```

## Result

- host_scalar_tests: pass
- CTest result: 8/8 tests passed
- IME asm executed on host: no
- host `bench_vmadot_microkernel`: `ime_status=not-built`

## Deterministic vectors

Covered:

- all zeros
- all ones
- A ramp, B ramp
- A alternating `-128/127`, B alternating `127/-128`
- random seed 0
- random seed 1
- random seed 12345
- nonzero initial C with `accumulate=true`

Generated vector artifact:

- `$LOG_DIR/artifacts/vmadot_test_vectors.json`
