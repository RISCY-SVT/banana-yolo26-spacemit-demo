# Package integrity report

All corruption and alias tests pass under normal and ASan/UBSan builds. A second exporter invocation produced an identical package tree and the same trusted manifest hash `0d3c3d49abdc8dd83857af223ea63bcb7a31058be4bcdb7cd7e6ccdf35659bac`. Every selected Conv recomputed weight sums and accumulator bounds successfully; all are int32-safe. Endianness policy is static little-endian rejection on unsupported hosts.
