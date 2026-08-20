# Stage65C-R1 append-only interpretation erratum

This file is an append-only clarification. It does not rewrite, delete, replace,
or silently supersede the original Stage65C-R1 report, classification, raw
evidence, metrics, or result packet.

```text
stage_id:
  BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001

original classification:
  stage65c-r1-recall-causality-inconclusive-frozen-a1-remains-blocked

original closure commit:
  d3afe14480ec2efbb2df9436deaa9022d631faa0

original final report SHA-256:
  714cf735be8f79ccda5bd9be1f87b0b6852e3ccbb0f935757af0e4868b34df84

original recovery report SHA-256:
  cb4334b509b4a960a279151e2c20229403fcdb70280b4f1bea934ac331e02583

original result packet:
  tree SHA-256: 8398831b147cc890436e968d830b14c0d5347ee5a24946b03156c66aa08b22e6
  files: 63
  bytes: 1983169
```

## 1. Host-restart clarification

Operator clarification:

```text
No distinct Windows, WSL, Ubuntu, VM, container, or orchestration-host reboot
occurred during Stage65C-R1.

The historical external Windows/WSL incident occurred several stages earlier.
Ordinary maintenance restarts performed between stages are environment
lifecycle events, not current-stage recovery events.
```

The current Stage did contain a board-side tooling defect:

```text
- the hash-smoke script used `index` as an awk scalar variable;
- the board awk implementation treated it as conflicting with a built-in;
- the incomplete smoke roots were isolated and excluded;
- the variable was renamed;
- the clean smoke passed;
- 100 in-session repeats and 10 clean session recreations passed.
```

Therefore references to a current Stage65C-R1 host restart are provenance
wording errors. They do not invalidate accepted model, task metric, bootstrap,
boundary, tail replay, determinism, or protected-state evidence.

Future stages may claim a current-stage host reboot only when backed by one or
more of:

```text
host boot identity before/after
OS/hypervisor boot-boundary evidence
current-stage event timestamp and operator attestation
current-stage interrupted process uniquely linked to the event
```

A partial command root alone is not proof of an operating-system reboot.

## 2. Intrinsic AR-large interpretation

Accepted full-val comparison:

```text
A1 CPU - B2 CPU AR-large:
  point = -0.003921852513

95% CI:
  [-0.006764457637, -0.001502683327]
```

Correct interpretation:

```text
a small negative model-intrinsic AR-large trade-off:
  statistically supported

a material intrinsic point loss beyond the predeclared -0.005 threshold:
  not confirmed
```

The original primary classification remains unchanged because it was generated
from the predeclared Stage rule.

## 3. A1-specific provider interaction

For:

```text
(A1_EP - A1_CPU) - (B2_EP - B2_CPU)
```

AR-large is:

```text
point:
  -0.001580895199

95% CI:
  [-0.006150002123, +0.003599398940]
```

The A1-specific provider penalty remains statistically inconclusive. No
SpacemiT EP correctness bug is claimed.

## 4. Historical point gate versus non-inferiority inference

The original Stage gate used:

```text
A1 EP - B2 EP AR-large >= -0.005
```

Observed point:

```text
-0.005502747712
```

Therefore the historical gate remains `FAIL`.

The 95% interval:

```text
[-0.009486292579, -0.001125071172]
```

crosses the non-inferiority margin `-0.005`. Under a hypothetical predeclared
three-way interval rule, the result would be `INCONCLUSIVE` against that margin.
This observation does not retroactively modify the Stage gate.

## 5. Selected-case causal scope

The 64 diagnostic cases were deliberately enriched for divergence.

The single-boundary splice metric reports:

```text
reduction of final-output numerical distance toward CPU
```

It does not directly report:

```text
population COCO recall recovered
```

Supported narrow result:

```text
confidence boundaries dominate selected-case numerical divergence;
P5 confidence is strongest on selected large-loss cases;
bbox boundaries are nearly inert;
the deterministic common tail amplifies score/rank/TopK differences.
```

No selected-case result is promoted into a population-level provider-causality
claim.

## 6. Thermal instrumentation debt

Some full-val raw roots contain zero-byte:

```text
thermal_before.tsv
thermal_after.tsv
```

These files are not thermal evidence and must not support a performance claim.

Future runner fix:

```bash
for temp_path in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "${temp_path}" ]] || continue
    printf '%s\t' "${temp_path}"
    cat "${temp_path}"
done
```

## 7. Output evidence roles

The companion `EVIDENCE_INDEX.yaml` distinguishes:

```yaml
input_evidence:
  stage65c_packet: ...
  frozen_models: ...
  common_tail: ...

output_evidence:
  tracked_stage: ...
  raw_stage: ...
  result_packet: ...
  shared_log: ...
  post_push_attestation: ...

excluded_or_partial:
  smoke_roots: ...
  reason: ...
```

## 8. Unchanged final disposition

```text
Stage65C-R1:
  accepted diagnostic evidence

A1:
  frozen research artifact
  not promoted

B2:
  vendor-lane universal control

performance / soak:
  not opened

XSlim:
  unchanged in this Stage

custom executor:
  unchanged

/data/ncnn:
  unchanged
```
