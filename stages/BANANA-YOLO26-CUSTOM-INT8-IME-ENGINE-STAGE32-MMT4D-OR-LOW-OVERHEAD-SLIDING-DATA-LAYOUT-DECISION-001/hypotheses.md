# Stage32 Hypotheses

H1: Stage31 direct/sliding failure is dominated by panel_build, not by `smt.vmadot1/2/3` compute.

H2: A low-copy sliding A-window design must reduce panel_build by >=5x (`38901.3 us` -> `<=7800 us`) before any direct/sliding integration is justified.

H3: If low-copy panel_build remains `>7800 us`, the `vmadot1/2/3` direct/sliding lane is rejected for now and MMT4D/threaded remains mainline.

H4: The integer dot signedness family may reduce correction overhead, but requires parser/disassembly/board/oracle proof before any implementation.

H5: MMT4D already uses IME through plain `smt.vmadot`; Stage32 must not describe MMT4D as non-IME.
