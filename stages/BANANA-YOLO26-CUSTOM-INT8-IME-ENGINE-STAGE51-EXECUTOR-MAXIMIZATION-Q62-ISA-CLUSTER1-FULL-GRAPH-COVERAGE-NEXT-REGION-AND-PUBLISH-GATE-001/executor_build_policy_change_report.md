# Executor build-policy change report

No governing ISA policy changed. The selected contract remains
`-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`.
Therefore no CMake preset, environment script, `k1x-env-overview.md`, or active policy text was
changed in Stage51.
