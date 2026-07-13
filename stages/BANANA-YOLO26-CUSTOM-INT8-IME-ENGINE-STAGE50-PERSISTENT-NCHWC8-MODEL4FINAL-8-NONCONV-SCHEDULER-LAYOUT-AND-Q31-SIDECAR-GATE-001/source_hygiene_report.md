# Source hygiene

Native build and 48/48 CTest pass; corrected ASan/UBSan build and 48/48 CTest pass; Python compile pass; full RISC-V build pass; board loader pass; readelf shows no RPATH/RUNPATH; board ldd resolves system libraries only. `git diff --check`, symlink scan, large-file scan, scoped secret scan, and scoped private-path scan pass. `/data/ncnn` retains its exact pre-existing HEAD, dirty paths, and file hashes. The first sanitizer CTest attempt raced the build and recorded 48 `Not Run` results; the preserved rerun passed all 48.
