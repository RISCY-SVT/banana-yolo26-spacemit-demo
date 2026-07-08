# Stage30 Summary RU

classification: `stage30-vmadot123-semantics-proven-but-no-speed-win`

Stage30 доказал микро-семантику `smt.vmadot1/2/3` для текущего K1X toolchain/board route. Все три инструкции собираются named asm, видны в objdump как символические инструкции, исполняются на CPU0-3 без SIGILL и проходят независимый scalar oracle на детерминированных fixtures.

Ключевая находка: `vmadot1/2/3` работают как shifted-M tile helpers над расширенным A-window, а не как готовая direct 3x3 Conv замена. Первый 32-byte A-window probe поэтому был неполным; после перехода на 8x8 A-window oracle стал точным.

Доказано:

- `smt.vmadot1`: parser/assembler/disassembly/board/oracle pass.
- `smt.vmadot2`: parser/assembler/disassembly/board/oracle pass.
- `smt.vmadot3`: parser/assembler/disassembly/board/oracle pass.
- CPU0/1/2/3: `status_failures=0`, `traps=0`, `validation_mismatches=0`.

Не сделано:

- direct Conv sidecar не принят и не интегрирован.
- full YOLO26 engine не реализован.
- model FPS, camera/full-image, COCO/mAP и production readiness не заявлялись.

Следующий шаг: отдельная Stage31 стадия для одного bounded expanded-A-panel direct/sliding 3x3 Conv кандидата на `/model.4/m.0/cv1/conv/Conv` с сравнением против текущего threaded MMT4D.
