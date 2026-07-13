# Q62 E2c decision

Status: selected. Parser, assembler, objdump, board execution, exact arithmetic, and vector CSR
restoration all pass. M12 plus the exact tail remains selected.

- model5 E1 to E2c: 4846.126106 -> 3658.164866 us (24.513626% lower).
- model4-final to model8 E1 to E2c: 27001.599444 -> 17828.345456 us (33.973002% lower).
- E2c p95 satisfies the no-regression gate.
