# Next-region architecture

The package extends the existing arena/schedule to 39 tensors and 36 operations. The region uses
shared immutable packed weights, the persistent CPU0-3 IME pool, explicit RVV MaxPool, direct
four-way Concat placement, exact Q62 E2c, and CPU4 controller only. Prepare/run/destroy are
separate; the timed path performs no allocation, file I/O, ORT call, conversion, or float Q/DQ.
