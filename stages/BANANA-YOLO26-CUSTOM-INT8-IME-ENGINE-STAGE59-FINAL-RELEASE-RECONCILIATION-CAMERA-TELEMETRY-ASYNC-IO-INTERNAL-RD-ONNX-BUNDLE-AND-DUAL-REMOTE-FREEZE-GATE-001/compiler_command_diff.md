# Compiler Command Difference

The neutral Stage57 and Stage58 endpoint builds used the accepted release flags:

```text
-march=rv64gcv_zvfh
-mabi=lp64d
-mtune=spacemit-x60
-funroll-loops
-O3
-DNDEBUG
```

The published Stage58 library was instead produced through the top-level
`scripts/build_cross.sh` path. Its toolchain supplied only
`-march=rv64gcv_zvfh -mabi=lp64d`; the top-level project did not pass the
specialized executor flag variable used by the standalone Stage57 build.
Consequently `-mtune=spacemit-x60 -funroll-loops` were absent from every hot
executor translation unit.

Stage59 fixes this at the target boundary. When
`Y26_K1X_OFFICIAL_RELEASE=ON`, the release object target itself carries all
four architecture/tuning flags. Configure fails if IME is disabled or the
target is not RISC-V. Host scalar tests remain available with the option off.
