# PMU method V2

Each pinned worker opens a cycles group leader and instructions member for its own task, then performs grouped reset, enable, exact operation, disable, and one PERF_FORMAT_GROUP read. Values remain unsigned and include time_enabled/time_running, event ID, TID, CPU, iteration count, binary identity, and package identity. No separately executed envelopes are subtracted.
