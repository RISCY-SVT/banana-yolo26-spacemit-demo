# Deploying The K1X INT8 Executor

All executables, libraries, packages, logs, and outputs belong on board NVMe
under `/data`. Do not deploy bulky artifacts to `/home/svt`, `/tmp`, or eMMC.

The release bundle deploys to:

```text
/data/k1x-yolo26-int8-executor/
```

A versioned optimized-research bundle may use a child such as
`/data/k1x-yolo26-int8-executor/stage56-optimized-research`; this keeps both
handoffs on NVMe without replacing the Stage52 functional reference.

`config/k1x-int8-executor-safe.conf` records the supported package root,
worker/controller assignment, scheduler, input surface, and detector output
schema. It is a reviewable handoff reference; the CLI still requires explicit
arguments so deployment automation cannot silently change execution policy.

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

The Stage56 exact operator profile is explicit and independent from wake policy:

```bash
set -a
source /data/k1x-yolo26-int8-executor/stage56-optimized-research/config/k1x-int8-executor-stage56.env
set +a
```

The dedicated-board frame-gated epoch-spin wake policy is opt-in and remains SCHED_OTHER:

```bash
Y26_STAGE53_SPIN_POOL=1 Y26_STAGE55_FRAME_GATED_SPIN=1 \
  /data/k1x-yolo26-int8-executor/stage56-optimized-research/bin/yolo26_k1x_int8 \
  --package /data/k1x-yolo26-int8-executor/stage56-optimized-research/package \
  --image /data/example/bus.jpg --input-mode image \
  --output-json /data/example/bus-stage53.json \
  --threads 4 --pin 0-3 --scheduler safe --verify
```

It keeps worker CPUs active only inside an inference active window and parks
them between frames. It still uses more process CPU during inference than the
condition-variable compatibility mode. Enable it only on a dedicated measured
workload.

The bundled benchmark wrapper accepts an optional fourth argument:
`compatibility`, `low-latency`, or `low-latency-dedicated`. All source the
Stage56 operator profile. `low-latency` enables frame-gated epoch-spin;
`low-latency-dedicated` additionally applies the reversible Stage56 O2 system
profile for the benchmark duration and restores it through an exit trap.

The O2 profile isolates CPU0-4 in cgroup v2 and moves movable IRQs, unbound
workqueues, selected services, and normal system slices to CPU5-7. It does not
change the boot command line, kernel, storage location, THP policy, or CPU
frequency ceiling. Apply and restore it explicitly when running outside the
benchmark wrapper:

```bash
root=/data/k1x-yolo26-int8-executor/stage56-optimized-research
state=$root/state/stage56-o2
$root/scripts/stage56-system-profile.sh apply "$state"
# Move only the intended executor process into /sys/fs/cgroup/y26-stage56-inference.
$root/scripts/stage56-system-profile.sh restore "$state"
```

The original boot entry and NVMe runtime remain selected. tmpfs and the bounded
eMMC copy did not clear the warm pure-model selection gate.

`smoke-test.sh` validates package identity, loader resolution, CLI execution,
and deterministic output before handoff. `uninstall.sh` removes only the
specified deployment root.
