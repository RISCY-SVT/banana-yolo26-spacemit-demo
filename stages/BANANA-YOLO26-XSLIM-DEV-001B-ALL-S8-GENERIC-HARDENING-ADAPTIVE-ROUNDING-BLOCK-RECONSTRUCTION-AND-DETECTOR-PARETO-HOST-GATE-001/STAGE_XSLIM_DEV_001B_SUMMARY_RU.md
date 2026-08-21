# XSLIM-DEV-001B: краткое резюме

Этап завершил generic hardening XSlim и реализовал детерминированные adaptive
rounding, block reconstruction и bias correction без изменения all-S8 QDQ
топологии. Режим без override побайтово воспроизвел B2. Все четыре новых
кандидата дважды воспроизводимы, сохраняют 812 Q/DQ, 0 QLinear, 0 UINT8,
0 FP16, 102/102 `kernel_shape`, шесть выходов и неизменный float tail.

На H500 лучший сигнал дал C2: mAP вырос на 0.010772 относительно B2, AP-large
на 0.022827. Однако вероятность выполнения ограничения AR-small составила
0.8361 при требуемых 0.90. C3 не прошел AP-medium и Pareto к A1, C4 ухудшил
AP-small/AP-medium, C5 не достиг порога mAP. Поэтому full val2017 не запускался,
новые кандидаты не создавались, а all-S8 PTQ reconstruction lane для YOLO26
закрыт.

Первичная классификация:

```text
xslim-dev-001b-all-s8-reconstruction-no-pareto-candidate-
close-this-ptq-lane
```

Изменения generic XSlim сохранены для ревью. Этап не запускал плату, не менял
релизы/теги, custom executor или `/data/ncnn` и не разрешает продвижение
runtime.
