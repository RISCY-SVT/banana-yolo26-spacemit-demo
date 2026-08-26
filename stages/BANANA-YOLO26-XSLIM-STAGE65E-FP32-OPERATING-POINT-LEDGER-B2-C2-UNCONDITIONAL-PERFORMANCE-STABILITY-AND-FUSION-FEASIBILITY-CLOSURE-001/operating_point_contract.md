# Stage65E operating-point contract

Counts are derived from `COCOeval.evalImgs` after category-wise COCO matching. For each score threshold, selected detections retain `dtIgnore`; TP/FP use `dtMatches` and `dtIgnore`, while FN is the set of non-ignored GT IDs not matched by a selected non-ignored detection. Crowd and area-ignore semantics use `gtIgnore`. MaxDets 100 and 300 are evaluated in separate COCOeval passes. The implementation passes an explicit crowd/ignore synthetic oracle and exact re-accumulation checks against every accepted aggregate metric.
