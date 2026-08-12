# Итоги Stage65B-R2

Классификация: `stage65b-r2-independent-selection-pass-upstream-branch-error-material-early-subgraph-r3-ready`.

- Независимый H500 выбрал `B2`; `B0` занял второе место.
- В текущем runner поверхности FP32 F0/F1/H8 полностью совпали на H500 и
  val2017, поэтому историческое расхождение FP32 относится к старому harness.
- D8 удалил только шесть финальных Q/DQ, но восстановил лишь
  `0.375` полного разрыва mAP: существенная ошибка накоплена выше
  по ветвям.
- Проверка seed/order/membership: `no-significant-aggregate-map-sensitivity-proven`.
- Для Vseed full-val не запускался: P(delta>0)=0.94 ниже заданного порога
  0.95. У Vdraw обнаружен отдельный membership-сигнал AP-small/AP-medium без
  доказанного изменения общего mAP.
- Единственный следующий маршрут: `R3-early-subgraph-localization-ready`.
- Плата K1X, SpacemiT EP, производительность и продвижение runtime не
  проверялись и не разрешены.
