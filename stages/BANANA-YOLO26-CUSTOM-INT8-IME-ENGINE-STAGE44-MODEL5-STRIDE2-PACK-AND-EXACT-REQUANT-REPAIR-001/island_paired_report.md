# Paired island and scaffold result

Final Path A (model4-only) is `510864.440692 +/- 396.409212 us`. Final Path B (model4-to-model5 R2a) is `515063.225590 +/- 245.618531 us`. Paired B-A is `+4198.784898 +/- 631.138511 us`, or `+0.821898%`.

Path B has zero model4-to-model5 materialized transpose. Its internal custom model4-to-model5 work is `49641.074778 us`; one entry and one exit adapter bring this to `64432.192022 us`. It saves about `18.63 ms` in the suffix relative to A, but custom model5/postactivation and adapter work cost more.

The pre-repair R0 island was worse by `9301.556860 us` (`1.815641%`). R2a narrows but does not reverse the loss. Therefore:

- model5 compute: negative versus resource-matched ORT;
- internal island: exact, no independent ORT win established;
- island with adapters: negative;
- paired full hybrid scaffold: negative.

Output0 differs between A and B on the accepted board-ORT/discontinuous-head diagnostic surface. It is not used as the integer custom correctness gate.
