# CPU Affinity Plan

Current policy:

```text
IME / smt.vmadot execution: CPU0-3 only
CPU4-7 IME execution: forbidden
OpenMP/all-core default dispatch: forbidden
```

Stage41 board selected-mode run used:

```text
taskset -c 0-3
affinity_ok=1
```

Future CPU4-7 use is allowed only for explicitly scoped non-IME work after proving no `smt.vmadot` executes there.
