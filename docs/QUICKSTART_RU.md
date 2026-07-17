# Быстрый запуск за десять минут

## 1. Проверка платы

```bash
uname -m                         # riscv64
test -d /data && test -w /data  # NVMe доступен
nproc                            # 8
```

## 2. Проверка релиза

```bash
export RELEASE=/data/releases/banana-yolo26-k1x-int8-executor/0.9.2-stage59-final-runtime
cd "$RELEASE"
sha256sum -c SHA256SUMS
bin/yolo26_k1x_int8 --version
```

Ожидаемая версия начинается с `0.9.2/K1X_INT8_V1_.../abi1`.

## 3. Healthcheck

```bash
bin/y26_k1x_healthcheck \
  package \
  fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  fixtures/bus_640_nchw_f32.bin \
  0xd43f5e018b415631
```

Ожидаемый фрагмент:

```text
output_hash=d43f5e018b415631
```

## 4. Обычный запуск

```bash
taskset -c 0-4 bin/yolo26_k1x_int8 \
  --package package \
  --image fixtures/bus_640_nchw_f32.bin \
  --input-mode preprocessed-f32 \
  --profile compatibility \
  --expected-manifest-sha256 fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  --verify-determinism --verify-known-fixture \
  --expected-output-hash 0xd43f5e018b415631 \
  --output-json /data/y26-output.json
```

## 5. Низкая задержка на выделенной плате

```bash
scripts/o2-system-profile.sh dry-run
scripts/o2-system-profile.sh run -- \
  taskset -c 0-4 bin/yolo26_k1x_int8 \
  --package package --image fixtures/bus_640_nchw_f32.bin \
  --input-mode preprocessed-f32 --profile low-latency-dedicated \
  --expected-manifest-sha256 fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  --output-json /data/y26-output.json
scripts/o2-system-profile.sh status
```

После выполнения `snapshot` и `cgroup` должны иметь состояние `absent`.

## 6. Сборка C-примера

```bash
export PKG_CONFIG_PATH="$RELEASE/lib/pkgconfig"
riscv64-unknown-linux-gnu-gcc -O2 \
  "$RELEASE/examples/c_api_consumer.c" \
  $(pkg-config --cflags --libs y26-k1x-int8-executor) \
  -o /data/y26-consumer
```

Для CMake используйте `-DCMAKE_PREFIX_PATH="$RELEASE"`. Подробности приведены
в `INTEGRATION_GUIDE.md`.
