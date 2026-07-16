# Release Target Dependencies

The shared library depends only on the expected C/C++ runtime, libc, libm, pthread surface, and loader. CLI RUNPATH is `$ORIGIN/../lib`; no absolute build path is present. The installed tree has no repository lookup dependency.

Measured CLI process-start plus dynamic-load mean: 13106.148 us (50 launches). Executor prepare mean: 849192.569 us (20 handles).
