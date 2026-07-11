# Stage 43: краткий итог

Классификация: `stage43-model5-exact-no-compute-win`.

Пакет oracle для model.5-model.8 и прямые изолированные профили на плате готовы. Для model.5 реализованы scalar и IME-маршруты с постоянным NHWC layout: между custom model.4 и model.5 нет materialized transpose. На восьми наборах входов host scalar, board scalar и board IME совпадают побайтно с семантическим host ORT 1.27 `ORT_DISABLE_ALL`. Все режимы FRM проходят.

Найдено важное ограничение oracle: host `ORT_ENABLE_ALL` на стрессовых uint8 входах использует x86-оптимизацию с pairwise int16 saturation. Она расходится с точной DQ-Conv-Q семантикой. Этот x86-артефакт в K1X kernel не воспроизводился; обе поверхности сохранены отдельно, tolerance не вводился.

Производительность отрицательная: exact custom model.5 Conv+activation `26579.802 us`, same-session board ORT `17862.861 us`. Custom путь медленнее на `8716.941 us` или `48.799%`. Поэтому full hybrid scaffold и model.6 не запускались.

Коммит и push не создавались: данная отрицательная классификация не входит в разрешенный список для commit. Следующий шаг: отдельный Stage44 для stride-2 pack и быстрого exact requant, без model.6 и без новых ISA.
