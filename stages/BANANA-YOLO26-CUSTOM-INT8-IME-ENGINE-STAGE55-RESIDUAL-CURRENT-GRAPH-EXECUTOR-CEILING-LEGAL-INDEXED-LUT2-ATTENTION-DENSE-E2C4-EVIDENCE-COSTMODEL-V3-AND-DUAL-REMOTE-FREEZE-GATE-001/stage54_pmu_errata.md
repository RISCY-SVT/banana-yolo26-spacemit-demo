# Stage54 PMU erratum

Stage54 CPU-wide prefix subtraction produced negative cycles/instructions and impossible IPC. Those historical rows are invalid for kernel conclusions. Stage55 uses in-process, per-worker grouped perf_event_open reset/enable/run/disable/read records and unsigned u64 values.
