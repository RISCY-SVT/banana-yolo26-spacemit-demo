# STAGE_SUMMARY_RU

classification: track-b-pass-yolo26-value-confirmed

Полный COCO val2017 был доступен локально и на board, поэтому был выполнен полный bbox COCOeval для YOLO26n через публичный SpacemiT ORT rt204 SpaceMIT EP.

Ключевой результат: YOLO26n подтверждает ценность по точности: `AP≈0.405`, что выше импортированного YOLO11 production reference `AP≈0.384`. Скорость vendor rt204 при этом ниже YOLO11 production INT8: лучший измеренный YOLO26 вариант FP16 keep-I/O даёт `395.546 ms / 2.528 FPS` в app full-image smoke и `397.128 ms / 2.518 FPS` среднее на full COCO generation workload.

Вывод: продолжать тяжёлую custom-engine работу можно только как R&D lane с отдельными gate. Track B подтверждает ценность YOLO26 по mAP, но не создаёт production/default-backend claim. vmadot1/2/3 допустим только как отдельная proof lane после human review и Stage28 structural Conv evidence.
