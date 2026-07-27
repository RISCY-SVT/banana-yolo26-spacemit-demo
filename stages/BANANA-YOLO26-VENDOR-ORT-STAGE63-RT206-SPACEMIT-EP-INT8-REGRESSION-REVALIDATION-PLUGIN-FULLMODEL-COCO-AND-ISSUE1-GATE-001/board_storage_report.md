# Board storage

All Stage63 runners, runtime copies, models, fixtures, profiles, outputs,
predictions, fault evidence, temporary files, and logs are under:

```text
/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE63-RT206-SPACEMIT-EP-INT8-REGRESSION-REVALIDATION-PLUGIN-FULLMODEL-COCO-AND-ISSUE1-GATE-001/
```

`/data` is `/dev/nvme0n1p1` (ext4, `rw,noatime`). The board root filesystem is
eMMC and is not used for project artifacts. Global core-dump configuration was
not changed; fault capture uses process-local `gdb -batch`.
