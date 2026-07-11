# Краткий итог Stage45

Классификация: `stage45-model-executor-codesign-recommended`.

Лучший стабильный полный board ORT на CPU0-3: `461603.297250 us`. Реалистичный
standalone M12xN16 `smt.vmadot` достиг `54.360135 GMAC/s`, но даже перенос этой
скорости на все `2.740154 GMAC` даёт `50.407 ms`
без Q/DQ, активаций, layout, attention и head. Неизменённый YOLO26n-640 не имеет
достоверного пути к 45 ms.

На 500 изображениях COCO mAP50-95: FP32 `0.446714`, semantic
INT8 `0.410683`, operational INT8 `0.373479`.
Текущая INT8 поверхность теряет 3.603 AP относительно FP32 уже на направленном
подмножестве.

Следующий этап: спецификация и подготовка K1X student 416/512 с distillation+QAT
и статическим resident-INT8 AOT executor. Это прогноз и план, не достигнутые FPS,
не production и не подтверждённая точность.
