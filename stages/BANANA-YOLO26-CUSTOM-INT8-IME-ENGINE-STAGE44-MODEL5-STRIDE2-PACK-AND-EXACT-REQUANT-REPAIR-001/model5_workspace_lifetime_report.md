# Model5 workspace lifetime

Stage44 adds an explicit workspace lifecycle contract:

1. `y26_model5_island_workspace_init()` zeroes storage and writes magic/version.
2. `prepare()` rejects a workspace without that contract before any release/free.
3. `release()` frees only an initialized workspace and resets it while preserving the lifecycle marker.
4. All tool/test callers initialize before prepare.

The focused test fills a workspace with non-contract bytes and verifies that prepare returns `INVALID_ARGUMENT` and release does not free those values. ASan+UBSan with leak detection passes. This prevents the previous prepare path from calling release on uninitialized pointers.

R0 and R2a continue to prepack weights and allocate persistent state at prepare time; no custom-owned hot-loop allocation was introduced.
