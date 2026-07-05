# Model4 Bottleneck Decision

Stage16A representative/full-shape branch-entry timing passed correctness and identified `conv_us` as the dominant selected-subset bucket:

- `conv_share_pct=79.8539`
- `activation_share_pct=19.4265`
- `merge_share_pct=0.711448`
- `pack_layout_share_pct=0`

Stage16 compact model4 C2f completion is also correct, but compact timing is not enough to claim model4 full-shape C2f performance. The next stage should avoid further compact-only expansion and should run Conv/IME roofline plus controlled cluster0 threading feasibility on representative/full-shape model4 boundaries before broader graph expansion.
