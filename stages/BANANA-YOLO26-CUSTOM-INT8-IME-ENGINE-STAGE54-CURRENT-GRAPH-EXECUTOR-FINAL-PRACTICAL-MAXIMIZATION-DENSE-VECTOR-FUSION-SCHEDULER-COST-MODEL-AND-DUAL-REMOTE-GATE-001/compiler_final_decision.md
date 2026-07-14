# Compiler decision

Retain `-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops -O3 -DNDEBUG`. C1 was byte-identical and C2/LTO trapped on board; no governing compiler policy or Codex skill change is justified.
