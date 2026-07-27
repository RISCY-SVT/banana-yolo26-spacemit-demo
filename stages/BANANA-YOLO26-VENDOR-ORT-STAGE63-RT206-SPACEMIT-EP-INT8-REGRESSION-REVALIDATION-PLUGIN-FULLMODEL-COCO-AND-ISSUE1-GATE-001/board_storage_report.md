# Board storage

All Stage63 runners, runtime copies, models, fixtures, profiles, outputs,
predictions, fault evidence, temporary files, and logs are under:

```text
/data/k1x-stage-runs/BANANA-YOLO26-VENDOR-ORT-STAGE63-RT206-SPACEMIT-EP-INT8-REGRESSION-REVALIDATION-PLUGIN-FULLMODEL-COCO-AND-ISSUE1-GATE-001/
```

`/data` is `/dev/nvme0n1p1` (ext4, `rw,noatime`). At final inventory the task
root contains 947 files and 474,355,071 bytes. Global core-dump configuration
was not changed; fault capture uses process-local `gdb -batch`.

One bounded storage exception occurred during a filtered provider diagnostic.
The vendor provider inherited an unintended board-home working directory and
emitted 109 transformed ONNX subgraphs there (432,902 bytes total). Every file
was hashed, moved to the task-local NVMe diagnostic tree, and removed from
eMMC. The final matching-file count on eMMC is zero.

The diagnostic and performance launchers now change into their task-local
output directory and remove `SPACEMIT_EP_DUMP_SUBGRAPHS` from the environment
unless a subgraph dump is explicitly requested. No model, runtime archive,
COCO prediction, core file, build tree, or benchmark log was retained on
eMMC.
