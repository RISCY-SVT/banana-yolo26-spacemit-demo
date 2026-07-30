# Fault disassembly

Both Stage64 faults are explicit `SIGABRT` terminations after provider
unsupported-format errors. The captured PC is in
`__pthread_kill_implementation`; no illegal instruction or provider opcode is
attributed. Raw GDB instruction context remains under the task-owned NVMe
fault-backtrace path referenced by `fault_pc_opcode.tsv`.
