# Stage65C-R1 final report

## Decision

- Stage: `BANANA-YOLO26-XSLIM-STAGE65C-R1-A1-CPU-EP-LARGE-RECALL-DIVERGENCE-AND-TERMINAL-BOUNDARY-CAUSAL-DIAGNOSTIC-001`
- Classification: `stage65c-r1-recall-causality-inconclusive-frozen-a1-remains-blocked`
- Publication classification: `not-applicable-research-diagnostic-no-publication`
- A1 disposition: frozen and blocked; no performance or promotion review is opened.
- B2 disposition: retained unchanged as the vendor-lane matched control.

Full val2017 confirms that frozen A1 improves mAP and AP-large over B2 on both
CPU and SpacemiT EP. The A1 EP versus B2 EP AR-large point delta is
`-0.005502747712`, narrowly outside the original `-0.005` non-regression
limit. Neither predeclared causal explanation is established: the A1 CPU
AR-large delta does not reach the model-intrinsic point threshold, and the
A1-specific EP difference-in-differences interval crosses zero. The historical
Stage65C failure therefore remains valid and A1 remains blocked.

## Recovery after restart

The execution host restarted while the board hash smoke was being introduced.
On recovery there was no stale Stage, ORT, SSH runner, evaluator, or bootstrap
process. The only incomplete board surface had a header-only status file; one
valid B2 CPU inference had completed but had never been accepted into a metric.
It and the first clean reproduction were isolated as raw recovery evidence.

The reproduced orchestration failure was a board `awk` portability issue:
`index` is not accepted as a scalar variable by that implementation. Renaming
the variable to `i` fixed status parsing. A new four-surface smoke, a 2-case x
4-surface x 100-run matrix, and the accepted 100-run plus 10 clean-session
recreation matrix then passed. No decision surface used an incomplete root.

## Immutable binding

- Stage65C packet: `27bfec346a38cf365754478ca386f4985303eb3d910f71726c7ec09f5432ebcd`,
  69 files, 116456 bytes.
- A1 deployable/inference/range-manifest SHA-256: `8fad9fa0e385f58da281d963c5e18b010c80c402dcbeed0b46e3ca3065d010f3`,
  `f7c5345f68cf79a5c3748274239a14cdaa59f77eac0425f7771694febaa24632`,
  `e9ce9a1e71005d60ad18213d8110fbf51d4ab9ceb8509d9786989685aa0f7e6f`.
- B2 deployable/inference SHA-256: `0e7040d4e8b1b2d08a4e36cec4c99dcea6d52294e04901d17dfce10725c6d617`,
  `40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853`.
- Common tail SHA-256: `18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3`.
- ORT 2.0.6 asset/core/EP SHA-256: `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6`,
  `93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3`,
  `dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae`.
- Board: `bf3`, serial `92262f3b0dc4`, boot ID
  `0a0691d1-7502-44c3-903b-444dba83c1d9`.

The four accepted H500 prediction hashes were revalidated exactly. All four
surfaces contain 500 images, zero failed images, and zero non-finite
predictions.

## H500 statistics

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | 0.444400 | 0.242054 | 0.511671 | 0.680073 | 0.426601 | 0.706287 | 0.836058 | 65522 |
| B2 EP | 0.443557 | 0.238171 | 0.515103 | 0.673229 | 0.427353 | 0.708195 | 0.832675 | 65462 |
| A1 CPU | 0.451640 | 0.236515 | 0.519109 | 0.692920 | 0.422599 | 0.704812 | 0.830037 | 62968 |
| A1 EP | 0.450262 | 0.234944 | 0.520752 | 0.684372 | 0.418267 | 0.706266 | 0.814953 | 64159 |

The shared 10,000-draw H500 bootstrap used seed `65008`, draw SHA-256
`2c35138af6b27b417b0d7661dd65138e2e9268da64df3ebddc295a8c4ecab5db`,
and replicate payload SHA-256
`3b1a3120438c4ee7345b79f5ec0ee5ecef399f812fa20c1f90e28dd1d4152c31`.

- A1 EP - B2 EP mAP: `+0.006705286704`, 95% CI
  `[+0.001441927711, +0.013035811297]`.
- A1 CPU - B2 CPU AR-large: `-0.006020938556`, 95% CI
  `[-0.010989412949, -0.001991944256]`.
- AR-large interaction: `-0.011700950194`, 95% CI
  `[-0.017502665508, +0.007436335412]`; inconclusive.

H500 reproduces the historical recall failure, but its interaction interval
does not isolate an A1-specific provider effect. Full val2017 remains the
classification authority.

## Object and boundary attribution

Object attribution uses `COCOeval.evalImgs`. The frozen diagnostic selection is
32 large-loss images, 16 small-loss images, and 16 matched controls; its SHA-256
is `a821ed3339889750a03054a9741590128cfb7c8c3d9c6ee3a7f503aaa8d810f3`.
Large-loss events are concentrated toward high IoU thresholds; `person` and
`train` are the most frequent categories in the A1 EP versus B2 EP large-loss
surface. Exact images, objects, categories, scores, ranks, IoUs, and per-class
tables are preserved in the compact TSV evidence.

All 64 cases x four surfaces produced exact six-boundary dumps and replayed
through the common tail without mismatch. A1-specific CPU/EP RMS interaction in
exported-qparam steps ranks:

1. P3 confidence: `1.455948`.
2. P5 confidence: `1.406874`.
3. P4 confidence: `1.284210`.
4. P3/P4/P5 bbox: `0.568423`, `0.535704`, `0.519339`.

For the 32 large-loss cases, replacing one A1 EP boundary by the corresponding
A1 CPU boundary recovers most for P5 confidence (`0.048313` mean tail-distance
recovery), then P3 confidence (`0.033949`) and P4 confidence (`0.025413`). Each
bbox splice recovers less than `0.0001`. The narrow mechanism classification is
`terminal-confidence-difference-with-tail-rank-amplification`: divergence is
already present at confidence boundaries, and the common tail amplifies small
changes through score ordering and TopK membership. This selected-case result
does not prove a population-level provider interaction.

## Determinism

Every B2/A1 CPU/EP surface had one stable output hash and one stable six-boundary
manifest across 100 repeated runs in one session and 10 clean session
recreations. Board replay, host replay, and repeat host replay of the common
tail were byte-identical. No surface was non-finite, collapsed, or
nondeterministic.

## Full val2017

| Surface | mAP50-95 | AP-S | AP-M | AP-L | AR-S | AR-M | AR-L | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 CPU | 0.365646 | 0.179324 | 0.419598 | 0.519849 | 0.371934 | 0.646361 | 0.790647 | 651439 |
| B2 EP | 0.366251 | 0.178465 | 0.419035 | 0.516178 | 0.370576 | 0.646619 | 0.786035 | 652039 |
| A1 CPU | 0.372955 | 0.178620 | 0.423706 | 0.542217 | 0.370553 | 0.644550 | 0.786725 | 625537 |
| A1 EP | 0.372366 | 0.176708 | 0.422663 | 0.539644 | 0.366031 | 0.644010 | 0.780532 | 637132 |

All four runs completed 5000/5000 images with zero failures, non-finite
predictions, or collapse. The shared bootstrap used 10,000 replicates, seed
`65009`, draw SHA-256
`b09ad4926f2356c9555760833a331d01b0fcdc5265b699112efabba702aaa92e`,
and replicate payload SHA-256
`e9e324ba4fc95608c0d76f0762eb1939138a0b1a3f27e785eeee07a6ba96d920`.

- A1 EP - B2 EP mAP: `+0.006115297854`, 95% CI
  `[+0.005106382926, +0.009085346056]`.
- A1 EP - B2 EP AP-large: `+0.023466137878`, 95% CI
  `[+0.017967398760, +0.027272817613]`.
- A1 EP - B2 EP AR-large: `-0.005502747712`, 95% CI
  `[-0.009486292579, -0.001125071172]`; the original point gate fails.
- A1 CPU - B2 CPU AR-large: `-0.003921852513`, 95% CI
  `[-0.006764457637, -0.001502683327]`; negative, but the predeclared
  model-intrinsic point threshold (`< -0.005`) is not met.
- mAP interaction: `-0.001193654975`, 95% CI
  `[-0.003082438315, +0.001421854648]`; inconclusive.
- AR-large interaction: `-0.001580895199`, 95% CI
  `[-0.006150002123, +0.003599398940]`; inconclusive.

Every AP/AR size interaction interval crosses zero. The full-val A1 EP versus
B2 EP contract is not satisfied because AR-large is below `-0.005`; therefore
the H500 sampling-artifact classification is also not satisfied.

## Closure

- No performance ABBA, long soak, model regeneration, PTQ, Policy B, XSlim
  mutation, custom-executor execution, runtime promotion, or publication was
  performed.
- The board retained its boot ID; no rollback command was required.
- Project writes to the eMMC-backed root filesystem: zero.
- Banana protected main, custom executor, XSlim branch/tag/release, and the
  accepted ncnn head/tree/diff/three dirty paths are unchanged.
- No Stage runner, ORT inference, evaluator, bootstrap, SSH runner, or test
  process remains active.

Human review is required before any further provider-numerics investigation or
performance review. This Stage authorizes neither.
