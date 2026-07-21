# Current Graph Freeze

## Frozen Scope

After Stage57 publication, branch `yolo26-custom-int8-engine` is frozen for the
unchanged YOLO26n-640 graph.

| Identity | Value |
|---|---|
| Integer contract | `K1X_INT8_V1` |
| Graph profile | `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` |
| Model SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` |
| Package manifest SHA-256 | `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be` |
| Prediction SHA-256 | `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda` |
| Known output hash | `0xd43f5e018b415631` |
| Frozen executor release | `0.9.0-stage57-final-handoff` |
| Stage59 camera maintenance release | `0.9.2-stage59-final-runtime` |
| Stage60M scheduler maintenance release | `0.9.3-stage60m-maintenance-runtime` |

The source release commit and the containing publication commit are recorded in
the release manifest, `current_graph_freeze_record.md`, and post-push remote
parity evidence. A tracked file cannot contain the hash of the commit that first
contains itself; the post-push result packet is the canonical exact local/GitHub/
GitLab parity record.

## Frozen Measured Surface

The final 0.9.2 library measured 134100.921 us total mean and 135724.150 us
p95 in a same-session 1,000-sample O2 comparison. The rebuilt Stage57 control
measured 133356.369 us, so 0.9.2 is within the required 1% performance-
equivalence gate after restoring the accidentally omitted X60 tuning flags.
The separate 13,500-run 0.9.2 soak measured 133381.666593 us mean,
135151.521 us p99.9, and 135853.000 us maximum. COCO val2017 remained 5000/5000
and byte-identical at mAP50-95 0.3707408944391919.

## Frozen Selected Routes

- NCHWc8 spatial-inner activation layout.
- CPU0-3 named `smt.vmadot` IME workers; CPU4 controller.
- Exact Q62/RNE E2c5 dense epilogue and exact attention MatMul C8 epilogue.
- Stage56 direct 1x1, P3 stride2 delivery, depthwise V2, lifetime-safe fusion,
  direct attention packing, and producer-direct head reduction.
- Compact C3 input and explicit RVV RGB copy.
- Condition-variable compatibility wake and frame-gated-spin low-latency wake.
- Original boot, NVMe `/data`, and optional reversible O2 system placement.

## Allowed Changes on This Branch

- correctness and security fixes;
- build/toolchain or board-kernel compatibility regressions;
- dependency regressions;
- documentation, packaging, and release maintenance.

## Requires Explicit Unfreeze and a Separate Project/Branch

- new current-graph performance research;
- Q31 or any integer-contract change;
- model, layout, resolution, student, training, or co-design work;
- a new runtime/vendor lane, boot/kernel profile, or CPU4-7 IME use.

## Limits

The freeze is an engineering maintenance boundary, not a mathematical proof that
no exact optimization exists. Remaining exact measured and theoretical reserves
are listed in `remaining_optimization_reserve_ledger_v5.tsv`. Q31 and model
changes are not maintenance reserves; they belong to separately authorized work.
