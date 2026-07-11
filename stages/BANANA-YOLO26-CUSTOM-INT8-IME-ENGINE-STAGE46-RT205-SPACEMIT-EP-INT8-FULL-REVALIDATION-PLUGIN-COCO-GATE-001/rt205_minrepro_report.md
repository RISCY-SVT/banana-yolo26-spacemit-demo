# RT205 minimal-reproducer result

The historical explicit `kernel_shape=[3,3]` Q/DQ Conv bug persists. RT205 also
changes failure handling from a catchable ORT exception to an uncaught abort.
The same graph without optional kernel_shape does execute as an EP subgraph and
matches CPU bytes, so EP registration alone is not the blocker.

RT205 introduces an independent regression: QLinearConv full/control paths core
dump, and the tiny QLinearMatMul control exits with SIGILL (132), while both pass
under RT204. This is a runtime regression, not a custom IME or new-opcode test.
