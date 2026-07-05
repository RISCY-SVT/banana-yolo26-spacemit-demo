# Threaded Conv Design

## Scope

The Stage18 sidecar integrates cluster0 threaded execution only for:

```text
/model.4/m.0/cv1/conv/Conv
shape: 1x80x80x32 -> 1x80x80x16
kernel: 3x3 stride1 padding1
primitive: smt.vmadot MMT4D
```

It does not alter the default Stage15/16 single-thread runners and does not create a graph scheduler.

## API

New explicit sidecar API:

```text
include/y26_k1x_threaded_conv.h
kernels/conv_threaded.cpp
```

Key entry points:

```text
y26_threaded_conv_create_spatial_rows(...)
y26_threaded_conv_run_ime_cluster0(...)
y26_threaded_conv_get_plan(...)
y26_threaded_conv_worker_affinity_ok(...)
```

## Threading Model

```text
thread_count: explicit 1, 2, 3, or 4
CPU mapping: worker i -> CPU i
allowed CPUs: CPU0, CPU1, CPU2, CPU3
forbidden IME CPUs: CPU4, CPU5, CPU6, CPU7
threading implementation: std::thread + pthread_setaffinity_np
OpenMP/all-core dispatch: not used
global thread pool: not used
```

Workers are persistent inside `Y26ThreadedConvWorkspace`. The measured run path uses barriers and does not spawn worker threads inside each Conv call.

## Partition

Primary partition:

```text
spatial row split over output H
```

Each worker owns:

```text
local Y26PrepackedConvWeights
local Y26ConvWorkspace
local raw accumulator buffer
local corrected accumulator buffer
disjoint final output row range
```

The halo policy is overcompute-and-discard at chunk boundaries because the existing Conv params expose symmetric `pad_h` rather than separate top/bottom padding.

## Default Policy

The default runner path remains single-thread. The threaded path is selected only by explicitly creating and running the Stage18 sidecar workspace.
