# Compiler policy change

No compiler-policy or Codex-skill change was made. The project experimental executor contract remains explicit `-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`. `-mcpu=spacemit-x60` was tested only with an explicit `-march` and was not promoted.
