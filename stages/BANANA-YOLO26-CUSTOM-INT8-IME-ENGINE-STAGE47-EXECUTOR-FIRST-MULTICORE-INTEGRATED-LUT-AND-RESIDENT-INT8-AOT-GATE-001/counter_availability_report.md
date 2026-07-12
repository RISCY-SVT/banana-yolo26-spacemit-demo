
# Counter availability

The Stage47 `perf_event_open` probe used task clock, cycles, instructions, cache,
branch, and context-switch events without direct `rdcycle`. Raw result:

```
event	status	errno	error	count
task_clock	unavailable	13	Permission denied	0
cycles	unavailable	13	Permission denied	0
instructions	unavailable	13	Permission denied	0
cache_references	unavailable	13	Permission denied	0
cache_misses	unavailable	13	Permission denied	0
branches	unavailable	13	Permission denied	0
branch_misses	unavailable	13	Permission denied	0
context_switches	unavailable	13	Permission denied	0
```

Wall clock remains the selection metric.
