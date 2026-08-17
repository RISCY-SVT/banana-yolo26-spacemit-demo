# Stage65C final report

## Decision

- Stage: `BANANA-YOLO26-XSLIM-STAGE65C-A1-VS-B2-K1X-SPACEMIT-EP-PLACEMENT-CORRECTNESS-COCO-PERFORMANCE-AND-STABILITY-GATE-001`
- Classification: `stage65c-a1-board-correctness-or-task-agreement-blocked`
- Publication classification: `not-applicable-research-evaluation-no-publication`
- Stop gate: `H500 fail`

A1 preserves B2 provider placement and reproduces a positive H500 mAP gain on
both CPU and SpaceMIT EP. It does not pass the complete board-candidate gate:
EP recall regresses beyond the allowed small/large limits and A1 CPU/EP task
metrics differ beyond the declared agreement limits. Full COCO, matched ABBA,
and 10,000-run soaks were therefore not opened.

## Evidence closure

- DEV-001A packet: `4be1455764a4ffa28cdf523c5ac1b0ec509be38b8c9a20792404ae2dd97e6d12`,
  90 files, 434031 bytes.
- Bounded Drive verification passed for the canonical tracked Banana/XSlim
  Stage trees, raw-evidence role, packet, shared log, and post-push evidence.
  Full remote tree-byte verification remains nonblocking mirror debt.
- Frozen A1 deployable/inference/tail SHA-256: `8fad9fa0...`, `f7c5345f...`,
  `18ffff41...`.
- Frozen B2 deployable/inference/tail SHA-256: `0e7040d4...`, `40ba6a7f...`,
  `18ffff41...`.
- ORT 2.0.6 archive SHA-256: `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6`.

## Runtime and placement

Board `bf3`, serial `92262f3b0dc4`, boot ID
`0a0691d1-7502-44c3-903b-444dba83c1d9`, Bianbu 2.2.1, kernel 6.6.63.
The bound runtime reports ORT `1.24.2+spacemit.a1` and SpaceMIT EP header
`2.0.6`; loaded core/EP SHA-256 values are `93bb7560...` and `dcc95030...`.

- Signed S8-QDQ Conv and MatMul controls: pass on CPU and EP.
- Official qgelu and independent XOR plugin controls: exact pass.
- Affinity smoke on CPU0, CPU0-3, CPU4, CPU4-7, and CPU0-7: pass.
- B2 and A1 CPU/EP session creation, profiled run, and repeat run: pass.
- Each EP session exposes one 925-node fused subgraph, one unique SpaceMIT
  inference node in the profile, and zero CPU inference events.
- A1 and B2 partition topology is equal; A1 adds no unexpected CPU fallback.
  The separate FP32 tail intentionally uses CPU.

Profile durations are diagnostic only and are not promotion timing evidence.

## Fixed fixtures

All 16 B2/A1 x CPU/EP x F0/bus/Zidane/canonical runs produced finite
`1x300x6` output with nontrivial scores/classes and no collapse. CPU and EP
outputs are not falsely claimed byte-identical; boundary statistics and task
sanity are preserved in the fixed-fixture reports.

## H500

| Surface | mAP50-95 | mAP50 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | 0.444400 | 0.611991 | 0.242054 | 0.511671 | 0.680073 | 0.426601 | 0.706287 | 0.836058 | 65522 |
| B2 EP | 0.443557 | 0.613930 | 0.238171 | 0.515103 | 0.673229 | 0.427353 | 0.708195 | 0.832675 | 65462 |
| A1 CPU | 0.451640 | 0.619034 | 0.236515 | 0.519109 | 0.692920 | 0.422599 | 0.704812 | 0.830037 | 62968 |
| A1 EP | 0.450262 | 0.616985 | 0.234944 | 0.520752 | 0.684372 | 0.418267 | 0.706266 | 0.814953 | 64159 |

All four surfaces processed 500/500 images with zero failure, non-finite
prediction, or score collapse.

Ten-thousand-replicate paired image-level COCO bootstrap (`seed=65007`):

- A1 EP - B2 EP mAP: `+0.006705286704`, 95% CI
  `[+0.001496037023, +0.012950850487]`, P(delta > 0) `0.9942`.
- A1 CPU - B2 CPU mAP: `+0.007240173651`, 95% CI
  `[+0.001690245937, +0.012582567456]`, P(delta > 0) `0.9952`.
- A1 EP - A1 CPU mAP: `-0.001378162000`, 95% CI
  `[-0.005699986137, +0.005291219905]`.
- A1 EP - A1 CPU AR-large: `-0.015083694578`, 95% CI
  `[-0.024781068838, -0.000838373386]`.

The EP comparison passes mAP and AP-small/medium/large thresholds, but fails:

- AR-small delta: `-0.009086253892` (limit `-0.005`).
- AR-large delta: `-0.017721888749` (limit `-0.005`).
- A1 CPU/EP mAP difference: `0.001378162000` (limit `0.001`).
- A1 CPU/EP AP-large, AR-small, and AR-large differences: `0.008548174752`,
  `0.004332208775`, and `0.015083694578` (limit `0.003`).

Board CPU mAP remains close to the accepted host result (A1 difference
`-0.00008837`, B2 difference `-0.00026560`). EP is lower than host by
`0.00146653` for A1 and `0.00110887` for B2. This is task-level evidence, not
prediction-byte equivalence.

## Downstream disposition

- Full val2017: `not-run-gated-by-h500-failure`.
- Matched B2/B2 noise floor and A1/B2 ABBA: `not-run-gated-by-h500-failure`.
- Steady inference, tail, two-stage, and pipeline promotion timing: unknown.
- 1,000/10,000-run stability and resource soak: unknown.
- Runtime promotion readiness: `blocked`.
- A1 disposition: retain as a failed board-gate research artifact; do not
  promote or regenerate inside this Stage.

## Integrity

The board retained its boot ID and system configuration; final temperature was
55-56 C with all CPUs at the pre-existing performance governor/1.6 GHz. No
rollback command was needed. Project writes to the eMMC-backed root filesystem:
zero. Banana protected main, custom executor, XSlim source/tag/release, and the
accepted `/data/ncnn` head/tree/three dirty paths are unchanged. No Stage runner,
bootstrap, evaluator, performance process, or soak process remains active.

Deferred XSlim API hardening remains documented and unimplemented.
