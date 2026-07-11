# Full COCO2017 accuracy

The mandatory host 2x2 matrix completed on all 5000 images:

| Surface | mAP50-95 | mAP50 |
|---|---:|---:|
| FP32 ORT_DISABLE_ALL | 0.401438855549 | 0.557787239518 |
| FP32 ORT_ENABLE_ALL | 0.401438842668 | 0.557787386415 |
| INT8 ORT_DISABLE_ALL | 0.372453424642 | 0.526269698607 |
| INT8 ORT_ENABLE_ALL | 0.333615160723 | 0.480129854852 |

FP32 optimization is effectively invariant (`-0.000000012881`).
Semantic INT8 loses `-0.028985430907` mAP50-95 relative to FP32,
and ORT_ENABLE_ALL introduces another `-0.038838263918` on the INT8 graph.

RT204 CPU full COCO mAP50-95: `0.374594101158669`.
RT205 CPU full COCO mAP50-95: `0.374594101158669`.
Their deltas versus fixed-host semantic INT8 are
`0.002140676516974971` and
`0.002140676516974971`.
The two package CPU prediction JSON files are byte-identical when both rows are
complete (`1dbc118383009fa68ddb1b786af68f56b38c473d8c03a23a782a38e4727d0b48`).
RT204 dataset mean inference/full-pipeline time is
`997.8373735757999 / 1018.6124350986 ms`;
RT205 is `996.6670820648 / 1016.7803087579999 ms`.
Both SpacemiT EP full-COCO rows are not runnable and are not encoded as zero.
No EP accuracy parity claim is possible, and no bootstrap EP delta is meaningful
without an EP output surface.

The fixed host semantic blank fixture has 28 rows at score >=0.001 (maximum
score 0.0167016685); this behavior is recorded rather than hidden. The structured
0/1 edge fixture has no row at that threshold. RT204/RT205 CPU exactness for
these fixtures is part of the fixed-fixture gate; the EP path cannot reach them.
