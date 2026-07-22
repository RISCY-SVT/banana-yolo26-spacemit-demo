# Source Access and Preferred Form

The preferred form for modification is the complete-source Stage62 archive. It
contains project C/C++ and assembly source, headers, CMake/toolchain files,
scripts, configuration, package/static-profile generators, tests, exact source
ONNX, all nine static ONNX profiles, prepared package trees, manifests, and
build instructions. COCO images are not included.

```text
stable R640: v0.9.3-r640 / d0e3611c8d99dfade049bd261cb557509222a456
Stage61 Q0:  stage61-q0-final / fa668ccaf7938336bd10313455ab81557b33e020
integrated:   v0.10.0-internal-rd.1 / see FINAL_SOURCE_HASHES.tsv
```

Compiler contract:

```text
-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60
-funroll-loops -O3 -DNDEBUG
```

Use `docs/BUILDING_K1X_INT8_EXECUTOR.md` and the archive-local
`BUILD_REPRODUCE.md`. Profile identities and hashes are in
`PROFILE_PROVENANCE.tsv`. The complete-source archive itself is the supplied
source form; this project does not operate a network source-offer service.
