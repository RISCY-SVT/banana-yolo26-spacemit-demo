# track_b_decision

classification: track-b-pass-yolo26-value-confirmed

## Answers

- Does YOLO26 vendor-ORT rt204 mAP justify continuing heavy custom-engine work? Yes, conditionally. Full COCO AP is about `0.405`, above the imported YOLO11 production reference `0.384006`, so the model has accuracy value. Heavy custom-engine work should remain gated and selected-lane only because vendor rt204 speed is still below YOLO11 production INT8.
- Is YOLO26 accuracy better/similar/worse than YOLO11 production reference? Better on this measured COCO bbox AP table: YOLO26 FP16/FP32 `AP≈0.4047`, `AP50≈0.571`, `AP75≈0.435`.
- Is YOLO26 rt204 speed acceptable as a fallback? Acceptable for R&D/validation fallback, not production replacement. FP16 keep-I/O app full-image smoke is `395.546 ms / 2.528 FPS`; full COCO generation mean is `397.128 ms / 2.518 FPS`.
- Should vmadot1/2/3 direct-conv proof lane be opened now, deferred, or rejected? Open only as a separate proof lane if the custom-engine Conv roofline remains structurally limited after current selected-cut repairs. Track B supplies model-value justification but not permission to integrate vmadot1/2/3 into the engine.
- Should vendor packets be sent now, and to which teams? Recommended after human review: send rt204 Q/DQ Conv clip-minmax packet to SpacemiT ORT/EP runtime team, and XSlim static PTQ blockers to XSlim/upstream tooling team. No packets were sent in this task.
