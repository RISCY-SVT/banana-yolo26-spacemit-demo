# Symbol and Version Report

The final maintenance build is ELF64 RISC-V LP64D, release version `0.9.2`,
SONAME `liby26_k1x_int8_executor.so.1`, and ABI version node
`Y26_K1X_ABI_1`. The shared object is 454464 bytes with 320996 bytes of
`.text`; the static archive is 2314834 bytes.

Exactly 15 C API functions are dynamically exported, all at
`Y26_K1X_ABI_1`:

```text
y26_build_info_init
y26_executor_copy_boundary
y26_executor_create
y26_executor_destroy
y26_executor_get_build_info
y26_executor_get_output
y26_executor_last_error
y26_executor_options_init
y26_executor_prepare
y26_executor_run_preprocessed
y26_executor_run_rgb
y26_executor_tensor_bytes
y26_executor_tensor_id
y26_executor_version
y26_status_string
```

There are no exported C++ or research symbols. Stage59 does not remove,
rename, or change the version of any prior ABI1 symbol. The only dynamic
dependencies are `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6`,
and the RISC-V dynamic loader. The relative `$ORIGIN` RPATH contains no host,
build-tree, or absolute runtime path.
