# Краткий итог Stage65C

- Классификация: `stage65c-a1-board-correctness-or-task-agreement-blocked`.
- Точные A1/B2 и ORT 2.0.6 привязаны по SHA-256; XSlim и защищенные проекты
  не изменены.
- S8-контроли, плагины, affinity, создание сессий, размещение EP и 16 fixed
  fixtures прошли.
- У A1 и B2 одинаковая форма partition: один fused subgraph, без нового CPU
  fallback; отдельный FP32 tail штатно работает на CPU.
- На H500 A1 EP улучшил mAP относительно B2 EP на `+0.006705`, 95% CI
  `[+0.001496, +0.012951]`.
- AP-small сохранился в пределах gate, AP-medium/large выросли, но AR-small
  (`-0.009086`) и AR-large (`-0.017722`) нарушили лимит `-0.005`.
- A1 CPU/EP agreement также не прошел по mAP, AP-large, AR-small и AR-large.
- Поэтому full COCO, matched ABBA и 10k soak корректно не запускались; A1 не
  готов к promotion review, B2 остается контрольным артефактом.
- На eMMC нет project writes; rollback не требовался; незавершенных Stage
  процессов нет.
