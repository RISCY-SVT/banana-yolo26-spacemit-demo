# Human decision options

Stage64 opens no automatic promotion path. The evidence supports these
separate human decisions:

1. Decide whether to send the sanitized issue update and tiny repro bundle to
   SpacemiT. No GitHub issue was modified by this stage.
2. Decide whether the official XSlim 2.1.1 split route deserves a later,
   separately authorized promotion review. Such a review must address the
   measured accuracy loss, unsupported-input crash containment, dependency
   licensing, and model distribution rights.
3. Provide or approve a calibration corpus independent of COCO val2017 before
   treating the measured mAP as calibration-generalization evidence.
4. Decide whether the direct-E2E `ReduceMax` mismatch warrants requiring a
   vendor release newer than 2.1.1, despite the prescribed split route working
   on 2.1.1.
5. Keep the accepted custom executor and R640 release as the default unless a
   later explicit promotion decision says otherwise.

Training, QAT, model redesign, custom-executor changes, and external model
publication were outside Stage64 and remain unauthorized.
