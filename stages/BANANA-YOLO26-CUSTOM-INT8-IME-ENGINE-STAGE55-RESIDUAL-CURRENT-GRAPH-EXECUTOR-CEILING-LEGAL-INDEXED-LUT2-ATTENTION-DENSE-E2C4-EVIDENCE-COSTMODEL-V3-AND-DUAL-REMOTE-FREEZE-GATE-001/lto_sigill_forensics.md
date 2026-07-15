# LTO SIGILL forensics

The reproduced Stage54 LTO binary traps in `run_rgb_stem_chunk` at raw instruction 0x4a42a357 (`vsext.vf4 v6,v4`), not in indexed RVV and not in an IME symbol. LTO remains candidate-specific rejected evidence; it is not a global compiler conclusion.
