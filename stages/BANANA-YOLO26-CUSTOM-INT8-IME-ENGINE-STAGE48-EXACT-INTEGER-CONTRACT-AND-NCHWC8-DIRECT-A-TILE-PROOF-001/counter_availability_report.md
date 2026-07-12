
# Counter availability

`perf_event_paranoid=2`; the unprivileged probe returned EACCES for every event.
One non-persistent `sudo` retry succeeded for task-clock, cycles, instructions,
branches, branch misses, and context switches; the PMU reported zero for the
generic cache events. The probe measures diagnostic work only, not the model5
worker threads. Consequently cycles-per-vmadot is unknown and wall clock is the
selection authority. No sysctl was changed.
