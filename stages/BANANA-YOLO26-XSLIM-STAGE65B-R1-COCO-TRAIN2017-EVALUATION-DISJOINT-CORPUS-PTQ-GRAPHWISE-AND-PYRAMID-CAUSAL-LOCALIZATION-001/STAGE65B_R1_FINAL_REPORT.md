# Stage65B-R1 final report

## Classification

`stage65b-r1-evaluation-disjoint-calibration-pyramid-hypothesis-not-supported-host-root-cause-remains-open`

Publication classification: `host-research-evidence-only-no-board-or-runtime-promotion`.

## Corpus

The official COCO train2017 archive was acquired and selectively extracted. Exact JPEG and canonical decoded-pixel overlap with val2017 is zero for all selected calibration and H500 images. Model metadata proves COCO training lineage, so this corpus is evaluation-disjoint but not training-independent.

Open Images O1 is `not-run-nonblocking`: official metadata was captured, while the official image object host was outside the launch allowlist and an allowlisted equivalent returned HTTP 403.

## PTQ and host gates

All B1-B6 deployable ONNX models are byte-identical across two clean seeded generations. Each passed ONNX checking, signed-QDQ/QLinear/UINT8/kernel-shape/six-boundary/tail-identity checks, the accepted fixed F0/bus, Zidane, and canonical-image semantics, and 100-image host score-collapse gates. F0 and bus intentionally share the accepted `real_bus_preprocessed` tensor; the JPEG and input-tensor identities are recorded in `fixed_fixture_identity.tsv`. The best full-COCO global host candidate is `B2`. Hybrid causality was run on `B2`, selected earlier by the frozen scout rule; both identities are recorded in `later_candidate_recommendations.md`.

## Accuracy

| surface | mAP50-95 | AP-small | AP-medium | AP-large |
|---|---:|---:|---:|---:|
| FP32 | 0.40473065112282053 | 0.19778857258539873 | 0.4414523593136039 | 0.586958794539891 |
| B0 | 0.35876850879267863 | 0.179113696685979 | 0.4176396163742976 | 0.5165509362490022 |
| B2 | 0.3658592288412378 | 0.18014685069413666 | 0.41995524974853976 | 0.5201067045733788 |

Full B1-B6, H0/H1/H3/H5/H6/H8, per-class, size-bin, prediction-hash, Graphwise, and boundary evidence is in the adjacent reports.

## Causality

`earlier-subgraph-or-tail-interaction` for candidate `B2` under the predeclared recovery thresholds. The result comes from full COCO boundary replacement, not correlation metrics alone.

## Host-reboot recovery

The host reboot interrupted only B3 run2 during blockwise runtime calibration.
Its 4,085-byte incomplete tree was isolated as non-decision fault evidence and
never resumed in place. A clean B3 run2 reproduced the run1 deployable ONNX
and normalized Graphwise identities, and all later host gates completed. The
preceding boot journal is unavailable and kernel-ring access is denied, so the
reboot cause remains `unknown`; no OOM, panic, power, or filesystem cause is
inferred. Final process inspection found no unfinished stage process.

## Scope

No board command, ORT 2.0.6 placement claim, performance/soak run, XSlim source/release mutation, targeted model generation, QAT, training, issue update, or runtime promotion occurred.

Git closure, protected-project invariance, and result-packet identity are
recorded in the adjacent closure reports and exported packet manifest.
