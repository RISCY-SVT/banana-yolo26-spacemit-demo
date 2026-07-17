# ABI1 Stage59 Contract

Stage59 is additive release maintenance. SONAME and ABI remain `1`; no public
structure layout or existing symbol meaning changed.

One executor handle owns one arena and worker pool. `prepare`, both run calls,
output reads, boundary reads, and tensor metadata reads share one non-blocking
busy guard. Concurrent calls return `Y26_STATUS_BUSY`; the two metadata
functions use their documented `-1`/`0` sentinels and set `last_error` to a busy
message. Independent handles remain independent.

`y26_executor_destroy()` is not a synchronization primitive. The caller must
join or otherwise stop every operation on that handle before destroy. Tests do
not intentionally invoke a destroy race because that remains caller-side
undefined behavior.

`y26_run_timing` remains unchanged. Camera schema v2 is application evidence,
not an ABI structure extension.
