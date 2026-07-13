# K1X INT8 Executor API

The public C ABI is declared in `y26_k1x_executor.h`. C++ exceptions are caught
inside the implementation and never cross this boundary.

```c
y26_executor* y26_executor_create(void);
y26_status y26_executor_prepare(y26_executor*, const char* package_dir,
                                const char* trusted_manifest_sha256,
                                const y26_executor_options*);
y26_status y26_executor_run_preprocessed(y26_executor*, const float*, size_t,
                                         float*, size_t, y26_run_timing*);
y26_status y26_executor_run_rgb(y26_executor*, const uint8_t*, int, int, int,
                                float*, size_t, y26_run_timing*);
y26_status y26_executor_get_output(const y26_executor*, float*, size_t);
void y26_executor_destroy(y26_executor*);
const char* y26_executor_last_error(const y26_executor*);
```

One handle owns one arena, prepared package, worker pool, and output buffer.
Calls on one handle must be serialized. Use one independently prepared handle
per concurrent stream. Caller-owned input and output buffers remain owned by
the caller and must remain valid for the duration of a call.

`run_preprocessed` accepts exactly `1x3x640x640` contiguous float32 RGB values
in `[0,1]`. `run_rgb` accepts an already-letterboxed contiguous 640x640 RGB
uint8 image. Both produce 300 rows of six float32 values:

```text
x1, y1, x2, y2, confidence, zero-based-class
```

Diagnostic tensor lookup/copy APIs are available only when prepare used
`Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES`. Boundary capture disables the optimized
resident core and is excluded from measured execution.

The ABI reports argument, package, runtime, and unsupported errors explicitly.
The string returned by `last_error` remains owned by the handle and is replaced
by a later call.
