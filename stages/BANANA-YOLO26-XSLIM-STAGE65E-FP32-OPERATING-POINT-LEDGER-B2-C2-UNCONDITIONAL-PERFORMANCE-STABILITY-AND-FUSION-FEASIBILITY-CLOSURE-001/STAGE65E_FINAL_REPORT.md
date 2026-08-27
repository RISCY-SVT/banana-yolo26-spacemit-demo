# Stage65E final report

Classification: `stage65e-vendor-passport-complete-performance-stability-and-fusion-feasibility-closed-human-profile-decision-required`

Publication classification: `not-authorized-not-attempted`

Stage: `BANANA-YOLO26-XSLIM-STAGE65E-FP32-OPERATING-POINT-LEDGER-B2-C2-UNCONDITIONAL-PERFORMANCE-STABILITY-AND-FUSION-FEASIBILITY-CLOSURE-001`

## Scope and immutable inputs

Stage65E did not retry or relax the historical Stage65D-R1 accuracy gate. It used the exact frozen FP32, B2, A1, C2 and common-tail bytes, kept XSlim and the custom executor read-only, used no camera, and ran the unconditional B2/C2 performance and stability passport after runtime/placement controls passed.

## Accuracy ledger

The frozen FP32 prediction re-accumulated to exact mAP50-95 `0.4018217950262668` with the accepted prediction hash. All ten frozen host/board surfaces matched their accepted aggregate metrics.

| Surface | mAP50-95 | AP-L | AR-S | AR-L |
|---|---:|---:|---:|---:|
| FP32_HOST | 0.401821795 | 0.582946286 | 0.385843025 | 0.802694409 |
| B2_HOST | 0.365859229 | 0.520106705 | 0.373046142 | 0.789256171 |
| A1_HOST | 0.372935154 | 0.541288514 | 0.373047868 | 0.785733487 |
| C2_HOST | 0.378508778 | 0.558723160 | 0.373183454 | 0.786853666 |
| B2_BOARD_EP | 0.366251038 | 0.516177822 | 0.370575559 | 0.786035235 |
| C2_BOARD_EP | 0.378850178 | 0.556125231 | 0.367130774 | 0.782956957 |

C2 host improves over B2 by `0.012649549` mAP and recovers `35.17%` of the B2-to-FP32 mAP gap. On board EP the mAP gain is `0.012599140`. The historical universal gate remains failed because the predeclared AR-small and AR-large point/interval requirements were not all met; C2 is therefore a higher-AP profile candidate, not a universal replacement.

At score 0.25, IoU 0.50, maxDets 100, area all:

| Surface | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| FP32_HOST | 17404 | 4836 | 18931 | 0.782554 | 0.478987 | 0.594247 |
| B2_HOST | 14441 | 2698 | 21894 | 0.842581 | 0.397440 | 0.540113 |
| A1_HOST | 13754 | 2648 | 22581 | 0.838556 | 0.378533 | 0.521607 |
| C2_HOST | 13558 | 2491 | 22777 | 0.844788 | 0.373139 | 0.517639 |
| B2_BOARD_EP | 14427 | 2731 | 21908 | 0.840832 | 0.397055 | 0.539398 |
| C2_BOARD_EP | 13361 | 2400 | 22974 | 0.847725 | 0.367717 | 0.512938 |

The complete census covers five score thresholds, two IoUs, maxDets 100/300, four area bins, and per-class rows using COCOeval `evalImgs` match/ignore arrays.

## Runtime, performance and stability

B2 and C2 each re-attested as one equal 925-source-node SpaceMIT partition with zero unexpected CPU inference events; the common 34-node FP32 tail remained intentional CPU work.

Matched C2/B2 ratios are inference median `0.998976`, inference p95 `0.985635`, two-stage median `0.998895`, and two-stage p95 `0.989936`. Performance decision: `pass`. The decision uses eight B2/B2 and eight C2/C2 process/session noise-floor blocks plus twelve order-balanced ABBA blocks.

Two earlier incomplete harness roots are explicitly excluded: one exposed an orphaned watchdog-sleep design defect and one exposed a normal `/proc` process-exit sampling race. The clean accepted root was fresh, complete and independent; neither partial root contributes a timing row. No reboot is inferred from either root.

The first read-only custom-context root is also excluded as a tooling-output collision: omitting `--output-json` caused the executor's `/dev/stdout` JSON writer to truncate the redirected benchmark stream. A clean v2 run separated JSON and produced the complete deterministic 5x100 timing grid; no model, package or executable bytes changed.

Both models completed reversed-order short soaks and ten clean-session 1000-run segments. B2 10k two-stage median is `113871.093` us; C2 is `113873.535` us. Stability decision: `pass`; stable output/resource contract: `pass`.

## Read-only application context

The accepted custom package remains `0.10.0-internal-rd.1`, contract `K1X_INT8_V1`, with model SHA `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`. Its fixed-input median is `129917.515` us for the custom pure executor and `132103.500` us to `1x300x6`. Same-boot timing is application-level context only because model lineage, export, quantization, runtime, affinity and output implementation differ. No direct speedup ratio is claimed, and no rebuild or source mutation occurred.

## Fusion feasibility

Stage65F opening condition: `not-met`. Exact accepted ORT-level rows are `B2:disable, C2:disable`; BASIC/EXTENDED/ALL changed six-boundary bytes and were rejected even where a point timing looked lower. Offline optimization is `unsupported`, I/O Binding is `unsupported`, and EPContext is `unsupported`. Shipped plugin APIs and XSlim YoloDecode were capability-audited without changing accepted artifacts. The measured tail share is an upper bound only; no exact-tail implementation gain was projected. See `fusion_opportunity_ledger.tsv`.

## Disposition

- B2 remains the universal vendor control.
- C2 remains the frozen, best accepted same-source INT8 AP/mAP artifact and may be considered only through an explicit application waiver after reviewing TP/FP/FN and recall costs.
- A1 remains historical.
- Vendor PTQ search and provider-numerics localization remain closed.
- No runtime, model, XSlim release, custom executor, camera path or fusion implementation was promoted.

Raw evidence: `/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-STAGE65E-FP32-OPERATING-POINT-LEDGER-B2-C2-UNCONDITIONAL-PERFORMANCE-STABILITY-AND-FUSION-FEASIBILITY-CLOSURE-001`. Shared log: `/data/ncnn-logs/ai-team/2026-08-26/2026-08-26_21-14-59__contcodex__BANANA-YOLO26-XSLIM-STAGE65E-FP32-OPERATING-POINT-LEDGER-B2-C2-UNCONDITIONAL-PERFORMANCE-STABILITY-AND-FUSION-FEASIBILITY-CLOSURE-001__stage65e-passport`. Result-packet identity is recorded by the packet's own manifest and post-push attestation to avoid a self-referential tracked hash.

Timestamp: `2026-08-27T01:45:10Z`.
