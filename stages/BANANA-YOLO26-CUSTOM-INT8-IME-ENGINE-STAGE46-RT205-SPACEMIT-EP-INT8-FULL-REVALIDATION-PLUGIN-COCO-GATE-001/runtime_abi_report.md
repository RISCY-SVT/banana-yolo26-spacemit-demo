# Runtime and ABI report

RT204 and RT205 use identical ORT C API headers (`9ed0d7054a4e74249467365b25b415d36f51a44a6349e2a994a1812e4723d1e2`), API version
24, and core SONAME `libonnxruntime.so.1`, but different core and EP bytes. Each
runner was compiled and linked against its matching package. The installed board
runners use only relative `$ORIGIN` RPATH entries; board `ldd` resolved the
intended core/EP package and stage-owned OpenCV libraries.

RT205 core reports build commit `9bb02204b`; RT204 reports `c178f12b2`. The core
version string remains `1.24.2+spacemit.a1`, while the EP header/package version
is 2.0.5. Header compatibility is therefore not treated as binary equivalence.

Both release-specific runners were built with the SpacemiT RISC-V GCC 14.3
toolchain, `-march=rv64gcv_zvfh -mabi=lp64d`, Release optimization, and the
matching package include/core/EP paths. `GetVersionString`, `GetBuildInfoString`,
SONAME, DT_NEEDED, symbol versions, runner/library hashes, `readelf`, and board
`ldd` outputs are preserved in the raw command ledger. H127 is a Python-wheel
semantic runtime and therefore makes no compile-time-header claim.

The runner passes `SPACEMIT_EP_INTRA_THREAD_NUM` explicitly and records every
filter, dump, debug, and plugin provider option actually supplied. Package
defaults that are not stated in installed documentation remain unknown rather
than being inferred from successful session creation.
