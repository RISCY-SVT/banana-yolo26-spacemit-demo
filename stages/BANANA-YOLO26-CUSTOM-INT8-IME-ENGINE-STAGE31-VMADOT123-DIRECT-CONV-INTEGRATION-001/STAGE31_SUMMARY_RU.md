# STAGE31_SUMMARY_RU

classification: stage31-direct-conv-correct-but-no-speed-win

Stage31 доказал, что `smt.vmadot1/2/3` по-прежнему корректно выполняются на CPU0-3, и что один реальный direct/sliding 3x3 Conv sidecar для `/model.4/m.0/cv1/conv/Conv` дает byte-exact результат.

Но speed gate не пройден. Direct/sliding sidecar занял `56980.9 us`, тогда как текущий MMT4D занимает `20544.9 us` в 1-thread режиме и `5437.09 us` в 4-thread режиме. Основная причина отрицательного результата - `panel_build_mean_us=38901.3`.

`vmadotn` остается не авторизован: проверенные мнемоники отвергнуты assembler. Unsigned/mixed signedness variants видны в toolchain, но в Stage31 только задокументированы, без внедрения.

Что доказано:

- Stage30 `vmadot1/2/3` replay: pass.
- Primary direct Conv correctness CPU0-3: pass.
- mismatches: 0.
- max_abs_diff: 0.
- Host CTest: pass.
- RISC-V cross build: pass.

Что не доказано:

- Direct/sliding Conv быстрее MMT4D.
- Secondary `/model.4/m.0/cv2/conv/Conv`.
- `vmadotn` semantics.
- unsigned/mixed signedness family semantics.

Это не full YOLO26 inference, не model FPS, не camera/full-image performance, не COCO/mAP и не production/default-backend readiness.
