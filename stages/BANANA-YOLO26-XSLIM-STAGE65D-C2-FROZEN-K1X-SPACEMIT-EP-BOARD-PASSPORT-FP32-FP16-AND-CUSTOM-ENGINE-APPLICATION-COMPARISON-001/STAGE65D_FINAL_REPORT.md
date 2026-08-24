# Stage65D Final Report

## Classification

```text
stage65d-frozen-c2-k1x-provider-agreement-fail-
diagnostic-only-retain-b2
```

Publication classification: `not-authorized-not-attempted`.

Stage ID: `BANANA-YOLO26-XSLIM-STAGE65D-C2-FROZEN-K1X-SPACEMIT-EP-BOARD-PASSPORT-FP32-FP16-AND-CUSTOM-ENGINE-APPLICATION-COMPARISON-001`.

## Input Closure

- Banana started at `e4b2d9622bd6db39e3b69ff9ba425e806a18b3ea`, tree `db0d370d927d96119af91e4e031af703f30c9e20`, with exact GitHub/GitLab-RD parity.
- XSlim remained read-only at `46d5d36bcb6979bab6567fb4fe62839689f1881c`, tree `1788779cd0887a1c8e6924cd63ad7d16d42f41ca`, version `2.1.2+riscy.2.dev2`.
- DEV-001C packet verification passed: tree `ce214eb6e906586ffc98d5da823d4406bf1ea627d5e8ae65a823e259efdb38f1`, 44 files, 269,690 bytes.
- Frozen B2, C2 and common-tail bytes matched every expected SHA-256. No model generation, qparam change, quantization, training or source mutation occurred.
- Banana protected main, the custom-executor tree and `/data/ncnn` retained their accepted identities.

The exact final Banana branch head, tree and dual-remote equality are recorded in `banana_commit_inventory.tsv`, `banana_remote_parity_final.tsv` and the packet post-push attestation.

## Board And Runtime

The bound board was `bf3`, serial `92262f3b0dc4`, running Bianbu 2.2.1 and kernel 6.6.63. Its boot ID remained `0a0691d1-7502-44c3-903b-444dba83c1d9` throughout the Stage.

The official SpaceMIT ORT assets matched:

| Asset | SHA-256 |
|---|---|
| ORT 2.0.6 archive | `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6` |
| ORT core | `93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3` |
| SpaceMIT EP | `dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae` |

All accepted S8-QDQ Conv/MatMul controls, official plugin smoke, independent plugin smoke and affinity arms passed. Unsupported U8 controls were isolated and produced the expected bounded failure classification.

All Stage artifacts remained on the NVMe-backed `/data` mount. The final eMMC project-path count was zero. No alternate runtime, OS profile or persistent system setting required rollback.

## Placement And Correctness

B2 and C2 each compiled to one SpaceMIT fused inference subgraph containing 925 source nodes. Their graph I/O and op census were equal, and profiling showed zero unexpected CPU inference events. The separate, exact common FP32 tail is intentional CPU work.

F0, bus, Zidane and canonical fixtures passed on B2 CPU, B2 EP, C2 CPU and C2 EP with finite `1x300x6` output, nontrivial scores/classes and no collapse.

Determinism passed for every exact surface:

- 100 repeats in one session produced one stable output and boundary-manifest hash.
- 10 clean session recreations produced the same exact per-surface output and boundary-manifest hash.

CPU and EP outputs were intentionally assessed by task and numeric contracts; byte equality across providers was not assumed.

## H500 Board Scout

All four surfaces processed 500/500 images with zero failures, non-finite predictions or score collapse.

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | 0.444399891 | 0.242054478 | 0.511670654 | 0.680072558 | 0.426600587 | 0.706287213 | 0.836057750 | 65,522 |
| B2 EP | 0.443556616 | 0.238170547 | 0.515102966 | 0.673228961 | 0.427353314 | 0.708195493 | 0.832675006 | 65,462 |
| C2 CPU | 0.454249137 | 0.240330378 | 0.514488043 | 0.701385842 | 0.423790874 | 0.706129235 | 0.834748531 | 64,199 |
| C2 EP | 0.451104894 | 0.237307265 | 0.518233200 | 0.691383826 | 0.427300764 | 0.706515988 | 0.831065663 | 63,651 |

The shared 10,000-draw bootstrap used seed `65010`; draw-matrix SHA-256 was `6fc033562e5a123ea0f7b90eb1ab4986eb95e38eb140e151f9bff5610992ccd2`.

C2 EP improved mAP over B2 EP by `+0.007548278134`, 95% CI `[+0.004328282638, +0.014640085099]`, with `P(delta>0)=0.9996`. Every predeclared C2-versus-B2 AP/AR size-bin point and probability gate passed.

The separate provider-agreement point gate failed:

- C2 EP versus C2 CPU absolute mAP difference was `0.003144243371`, above the `0.002` limit.
- C2 EP versus C2 CPU absolute AP-large difference was `0.010002015827`, above the `0.005` limit.

The B2-controlled difference-in-differences did not prove a model-specific provider bug:

- mAP interaction point `-0.002300968317`, 95% CI `[-0.006525143092, +0.001806707155]`: inconclusive.
- AP-large interaction point `-0.003158418669`, 95% CI `[-0.012375158575, +0.006122290589]`: inconclusive.
- AR-large interaction point `-0.000300123423`, 95% CI `[-0.002316717673, +0.001628732517]`: provider-neutral under the frozen rule.

The prediction-count interaction is preserved as a descriptive material shift. It is not classified as a provider correctness defect without task-level causal proof.

## Closed Conditional Gates

Because the H500 provider-agreement gate failed, the Stage stopped the conditional route exactly as specified:

- full board val2017: `not-run-gate-closed`;
- matched B2/C2 performance and noise floor: `not-run-gate-closed`;
- B2 1k, C2 1k and C2 10k soaks: `not-run-gate-closed`;
- same-boot custom-engine execution: `not-run-gate-closed`.

No latency, thermal-soak, stability or full-val board conclusion is made for C2.

## Precision References

The exact reconciled FP32/H8 host reference was bound to model SHA-256 `72eb6136b41104753c53b8e13aeff50e7961c4cefba79e50b70894cbd169f8d8`, prediction SHA-256 `b9ff8fa19cba9682970d8e932f3318cdf5833094ab22256a24062019309b5b2a`, common-tail SHA-256 `18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3`, and host full-val mAP `0.4018217950262668`.

No comparable frozen FP16 artifact was available. FP16 is classified `not-run-no-comparable-frozen-artifact` and no historical cross-surface value was substituted.

## Custom-Engine Application Reference

The currently accepted custom package was resolved read-only as `0.10.0-internal-rd.1`, contract `K1X_INT8_V1`, source-model SHA-256 `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`, and accepted full-val mAP `0.3707408944391919`.

Its comparison with C2 remains an application-level cross-surface reference because model export, quantization and runtime surfaces differ. It is not an engine-only, quantizer-only or backend-superiority comparison. No custom-executor binary was rebuilt or executed in Stage65D.

## Disposition

- C2 preserves the B2 placement topology and shows a statistically supported H500 EP accuracy gain, but remains diagnostic-only because the frozen CPU/EP agreement point gate failed.
- B2 remains the vendor-lane universal control.
- A1 remains a historical frozen research artifact.
- The accepted custom engine remains unchanged as a cross-surface application reference.
- Runtime promotion is not ready and was not authorized.

Human review may retain B2, authorize a separate frozen-C2 provider-difference diagnostic, authorize an application-specific C2 profile after agreement is resolved, or separately authorize head-only QAT, model/executor co-design, or a same-source K1X_INT8_V2 comparison.

Evidence closure timestamp: `2026-08-24T15:14:38Z`.
