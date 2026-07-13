# Integer oracle report

Python arbitrary-precision package outputs, portable C++ scalar, board scalar, and final board IME agree byte-for-byte at all 32 boundaries for F0-F7. Tie, saturation, bound, tail, alias, and package-integrity tests remain exact. RNE/RTZ/RDN/RUP/RMM produce the same integer bytes and restore ambient FRM. IME execution is confined to CPU0-3; CPU4 is controller-only; no SIGILL occurred. Legacy float-QDQ ORT is not the integer authority.
