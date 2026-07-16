# K1X INT8 Executor C API

The stable colleague-facing interface is C ABI1 in `y26_k1x_executor.h`.
Version 0.9.0 exports only these C symbols from SONAME 1:

```c
void y26_executor_options_init(y26_executor_options *);
const char *y26_status_string(y26_status);
y26_executor *y26_executor_create(void);
y26_status y26_executor_prepare(y26_executor *, const char *, const char *,
                                const y26_executor_options *);
y26_status y26_executor_run_preprocessed(y26_executor *, const float *, size_t,
                                         float *, size_t, y26_run_timing *);
y26_status y26_executor_run_rgb(y26_executor *, const uint8_t *, int, int, int,
                                float *, size_t, y26_run_timing *);
y26_status y26_executor_get_output(const y26_executor *, float *, size_t);
int y26_executor_tensor_id(const y26_executor *, const char *);
size_t y26_executor_tensor_bytes(const y26_executor *, int);
y26_status y26_executor_copy_boundary(const y26_executor *, int, uint8_t *, size_t);
void y26_executor_destroy(y26_executor *);
const char *y26_executor_last_error(const y26_executor *);
const char *y26_executor_version(void);
```

Call `y26_executor_options_init()` before setting options. The default is four
workers on CPU0-3, controller CPU4, SCHED_OTHER, condition-variable wake, and no
boundary capture. Set `wake_policy=Y26_WAKE_FRAME_GATED_SPIN` for the public
low-latency wake profile.

The executor accepts exactly `1x3x640x640` float32 RGB NCHW input or already
letterboxed 640x640 RGB8. Output is 300 float32 rows of
`[x1,y1,x2,y2,confidence,class]`.

One handle owns one arena and worker pool. Calls on one handle must be serialized;
concurrent use returns `Y26_STATUS_BUSY`. Independent handles are supported.
Preparing twice and running before prepare return `Y26_STATUS_INVALID_STATE`.
Null, undersized, overlapping, and invalid-topology inputs are rejected.

The package prepare call requires the expected immutable manifest SHA. The loader
also binds schema, contract, graph profile, layout, and model SHA. C++ exceptions
never cross the ABI.

The internal C++ header is not installed by default. It is available only through
the explicit development/research CMake option and has no stable ABI promise.

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for a complete compilable
example and installed CMake/pkg-config use.
