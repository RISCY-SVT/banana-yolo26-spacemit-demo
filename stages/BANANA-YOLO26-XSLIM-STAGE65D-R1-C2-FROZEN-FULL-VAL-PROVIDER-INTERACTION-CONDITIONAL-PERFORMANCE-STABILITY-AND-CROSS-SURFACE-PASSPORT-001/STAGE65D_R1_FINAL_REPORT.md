# Stage65D-R1 final report

Classification: `stage65d-r1-frozen-c2-full-val-task-fail-retain-b2`
Publication: `not-authorized-not-attempted`
Stage: `BANANA-YOLO26-XSLIM-STAGE65D-R1-C2-FROZEN-FULL-VAL-PROVIDER-INTERACTION-CONDITIONAL-PERFORMANCE-STABILITY-AND-CROSS-SURFACE-PASSPORT-001`

## Identity and placement

The exact frozen B2/C2 inference models and common tail were used with SpaceMIT ORT 2.0.6. B2 and C2 each produced one equal 925-source-node fused SpaceMIT subgraph, six outputs, and zero unexpected CPU inference events. Bounded controls and F0 CPU/EP fixtures passed without non-finite output or score collapse.

## Full val2017

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | 0.365646337229 | 0.179323688060 | 0.419598143597 | 0.519849482899 | 0.371934110510 | 0.646361304477 | 0.790646745755 | 651439 |
| B2 EP | 0.366251037755 | 0.178464635293 | 0.419034700109 | 0.516177822164 | 0.370575558886 | 0.646618512013 | 0.786035234913 | 652039 |
| C2 CPU | 0.378625790211 | 0.178835569319 | 0.425950951488 | 0.558499516351 | 0.370355738654 | 0.645939798038 | 0.788051671109 | 638556 |
| C2 EP | 0.378850178125 | 0.178210773539 | 0.425354181318 | 0.556125230739 | 0.367130773892 | 0.645575537507 | 0.782956956986 | 633640 |

C2 EP minus B2 EP mAP50-95 is `0.012599140370` with 95% CI `[0.011641520920459936, 0.015099894258865584]` and `P(delta>0)=1.0`. AP deltas (S/M/L) are `-0.000253861755`, `0.006319481209`, `0.039947408575`. AR deltas (S/M/L) are `-0.003444784993`, `-0.001042974506`, `-0.003078277927`.

The predeclared task gate is `fail`. AR-small fails both its `-0.003` point guard (`-0.003444784993`) and `-0.005` lower-CI guard (`-0.006213496738686553`). AR-large misses the point guard by `0.000078277927` while its CI guard passes. This is a task-contract failure despite the material mAP/AP-large gain.

Prediction SHA-256 values: B2 CPU `c903721d880b1df599c6912455aa39106d94a2be2cd2ad226cce59fbdae28745`, B2 EP `edba82a970a95b4e13d194044573fadccebe831f98527116d1ca9a74b00eab39`, C2 CPU `186e53676f21f290e08f305aa78ad12031a3c7478698cb92535d881b8709dad5`, C2 EP `3a805d63c1e8e9ac05d843a2da87d6238f4ec6b52d3428e647ab6071f240e11a`. Every surface completed 5000/5000 with zero runner/evaluator failures, non-finite predictions, or collapse.

## Provider diagnostic

Population difference-in-differences contains `10` provider-neutral metrics, `2` inconclusive metrics, and `0` material metrics; overall status is `inconclusive`. mAP interaction is `-0.000380312612` with 95% CI `[-0.001993312583926846, 0.0008009431371904375]` and is provider-neutral. AR-small interaction is `-0.001866413136` with CI `[-0.005418503114108323, 0.0026728849662950483]`; AR-large is `-0.000483203281` with CI `[-0.0026884782287224076, 0.002032057440487601]`.

Score/rank analysis found deterministic CPU/EP sensitivity in both models, but no material C2-specific population interaction. It does not establish a provider bug, exact rounding mode, or LSB-level cause. EP-aware tuning is `not-justified-by-current-full-val-evidence`.

## Conditional passport

Matched performance: `not-run-task-gate-closed`. Stability: `not-run-task-gate-closed`. Custom-engine application context: `not-run-task-gate-closed`. These gates were correctly not opened after task failure; they did not fail experimentally. No camera work was run.

## Disposition

C2 remains frozen diagnostic evidence and is not promotion-ready. B2 remains the vendor universal control. XSlim, the custom executor, protected main and `/data/ncnn` are unchanged; board eMMC project writes are zero. No runtime default or persistent board state required rollback.
