# Stage59 Post-Stage60 Scheduler Erratum

This is an append-only correction. No Stage59 report or raw artifact is rewritten.

Stage59's arithmetic and release-identity closure remains valid: `K1X_INT8_V1`, the R640 full-graph profile, model/package identities, fixed output hash, all 215 exact boundaries, full-COCO result, and prediction SHA-256 are unchanged.

The later Stage60 resolution sweep exposed two liveness defects that were not known when Stage59 described the selected release as having no remaining defect:

1. a frame-gated `WorkerPool` worker could park after the controller's wake notification, leaving a required dispatch permanently incomplete;
2. threaded-convolution readiness could be published between the creator's predicate evaluation and condition-variable sleep because publication did not hold the predicate mutex.

Release 0.9.3 backports only the proven synchronization repairs. It does not alter arithmetic, layout, dispatch selection, package/profile identity, public ABI, camera policy, model assets, or quantization. Stage59 correctness and COCO conclusions therefore remain valid; its broader no-defect wording is superseded only for scheduler/startup liveness.
