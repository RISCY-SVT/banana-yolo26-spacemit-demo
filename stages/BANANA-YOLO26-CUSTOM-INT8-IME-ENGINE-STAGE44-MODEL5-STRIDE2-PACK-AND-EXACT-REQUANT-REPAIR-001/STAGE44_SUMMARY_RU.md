# Stage44: краткое резюме

Классификация: `stage44-model5-exact-no-net-win`.

Честный baseline изменил оценку: isolated board ORT model5 с intra4 работает за `11701.121842 us`, а не за исторические one-thread `17862.861218 us`. Лучший неизмененный custom R0 использует три worker.

Обязательный pre-repair island замерен: model4-to-model5 R0 медленнее model4-only на `9301.556860 us` (`1.815641%`). Реализован один ограниченный R2a stride-2 chunk fastpack. Он точен на F0-F7, проходит workers1-4 и FRM sweep. В ABBA microgate R2a улучшает R0 на `478.655 us` (`1.94291%`), но остается на `106.4537%` медленнее ORT intra4.

Финальный hybrid scaffold с R2a также проигрывает: `515063.225590 us` против `510864.440692 us`, разница `+4198.784898 us` (`+0.821898%`). Поэтому model5 tuning lane остановлен. R2a остается только экспериментальным non-default доказательством.

Следующий шаг: отдельный upper-bound/roadmap gate для объединенного model5-6 island без автоматической реализации model6.
