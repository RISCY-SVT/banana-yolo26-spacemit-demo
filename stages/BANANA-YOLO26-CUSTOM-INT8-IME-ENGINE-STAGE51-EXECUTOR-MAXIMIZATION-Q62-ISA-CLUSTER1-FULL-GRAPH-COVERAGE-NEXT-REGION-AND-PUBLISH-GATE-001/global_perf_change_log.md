# Global perf change log

`perf_event_paranoid` was changed temporarily from 2 to -1 and `kptr_restrict` from 1 to 0 for
the bounded PMU helper run. Both were restored to 2/1 before Stage51 completion. No persistent
sysctl file, capability, boot option, or global perf installation was created.
