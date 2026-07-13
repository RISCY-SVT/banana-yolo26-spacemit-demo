# Deploying The K1X INT8 Executor

All executables, libraries, packages, logs, and outputs belong on board NVMe
under `/data`. Do not deploy bulky artifacts to `/home/svt`, `/tmp`, or eMMC.

The release bundle deploys to:

```text
/data/k1x-yolo26-int8-executor/
```

Typical flow:

```bash
scripts/k1x-int8-executor/deploy.sh \
  svt@banana \
  /data/releases/banana-yolo26-k1x-int8-executor \
  /data/k1x-yolo26-int8-executor
ssh svt@banana \
  /data/k1x-yolo26-int8-executor/bin/yolo26_k1x_int8 \
  --package /data/k1x-yolo26-int8-executor/package \
  --image /data/example/bus.jpg \
  --input-mode image \
  --output-json /data/example/bus.json \
  --threads 4 --pin 0-3 --scheduler safe --verify
```

The default worker policy is CPU0-3 plus a CPU4 controller. The runtime must
not execute IME instructions on CPU4-7. Use `--scheduler rr20` only on a
dedicated lab board with sufficient privileges, a watchdog, and a cleanup
path. It is not the default.

`smoke-test.sh` validates package identity, loader resolution, CLI execution,
and deterministic output before handoff. `uninstall.sh` removes only the
specified deployment root.
