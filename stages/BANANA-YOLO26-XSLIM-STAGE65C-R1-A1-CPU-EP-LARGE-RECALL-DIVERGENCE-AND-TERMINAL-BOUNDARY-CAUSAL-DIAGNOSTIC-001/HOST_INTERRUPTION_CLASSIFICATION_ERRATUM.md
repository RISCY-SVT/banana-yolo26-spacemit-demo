# Host Interruption Classification Erratum

This append-only note does not alter the original Stage65C-R1 report,
classification, metrics, raw evidence, or result packet.

No distinct Windows, WSL, Ubuntu, VM, container, or orchestration-host reboot
occurred during Stage65C-R1. The current-stage defect was board-side awk
portability: the hash-smoke script used `index` as a scalar variable, which the
board awk implementation treated as conflicting with a built-in. Incomplete
smoke roots were isolated and excluded, the variable was renamed, and the clean
smoke plus 100 in-session repeats and 10 clean session recreations passed.

The historical Windows/WSL incident occurred several stages earlier. A partial
command root alone is not evidence of a current operating-system reboot.
