# Итог DEV-002

Классификация: `xslim-dev-002-riscy2-human-documentation-release-and-vendor-ptq-closure-complete`.

C2 зафиксирован отдельным TIER-1 профилем с повышенным AP. Исторический универсальный FAIL не изменен: B2 остается основным контролем и rollback. Для C2 перед применением обязателен отдельный подбор score threshold, поскольку при принятой точке он уменьшает FP, но увеличивает FN.

XSlim оформлен как пригодный для человека downstream-инструмент: добавлены установка, quick start, справочник конфигурации, настройка точности, честные границы reconstruction, K1X SpaceMIT S8-QDQ профиль, полный YOLO26 cookbook, диагностика, ограничения, provenance и русские руководства. 128 ссылок и 106 исполняемых фрагментов прошли проверку; 58 фрагментов корректно классифицированы как неполные примеры.

Релиз `2.1.2+riscy.2`, тег `v2.1.2-riscy.2`, commit `80204be2...` опубликован на GitHub и GitLab. Все восемь assets и release notes сверены после скачивания побайтно. Wheel и sdist установлены в чистые окружения и прошли import/version/CLI/config/pip/uninstall. PyPI не использовался.

Pytest прошел 212 тестов и 65 subtests без ошибок и неперехваченных warnings. Ruff, shellcheck, compileall, целевой strict mypy, Banana tooling, package/SBOM проверки прошли. Наследованный whole-tree mypy debt (2877 ошибок) и неполный per-file REUSE честно сохранены как долг, а не названы успешными.

Vendor PTQ, provider-numerics и текущая fusion-ветка закрыты. Full BRECQ/QDrop отложены. Возможный model/executor co-design требует отдельного разрешения и здесь не запускался. Protected main, custom executor и `/data/ncnn` не изменены; board не запускалась.
