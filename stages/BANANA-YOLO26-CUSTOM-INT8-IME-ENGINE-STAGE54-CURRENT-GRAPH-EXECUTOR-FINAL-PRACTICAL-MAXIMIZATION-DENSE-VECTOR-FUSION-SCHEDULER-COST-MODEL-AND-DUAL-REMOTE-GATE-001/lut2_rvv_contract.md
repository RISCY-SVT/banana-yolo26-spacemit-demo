# LUT2 RVV contract

The candidate forms exact u16 indices left*256+right and requests indexed byte loads from the package 65536-byte LUT. The local parser and objdump accepted it, but the board trapped with SIGILL; the exact scalar direct-physical route remains selected.
