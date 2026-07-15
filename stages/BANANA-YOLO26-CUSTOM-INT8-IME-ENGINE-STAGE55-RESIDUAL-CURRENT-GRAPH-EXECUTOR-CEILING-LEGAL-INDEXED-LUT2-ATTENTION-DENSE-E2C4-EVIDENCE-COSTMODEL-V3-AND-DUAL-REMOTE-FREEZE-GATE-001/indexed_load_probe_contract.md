# Legal indexed-load probe contract

Every case is child-isolated under SA_SIGINFO, uses a scalar oracle, records vtype/vill/vl/vstart/vcsr, and captures PC/opcode/symbol/CPU on a trap. The corrected e8,m1 LUT2 route builds u16 indices under e16,m2, restores data vtype e8,m1, and uses an even-aligned EMUL2 group.
