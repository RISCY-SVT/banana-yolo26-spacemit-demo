# Missing Stage43 island measurement

Path A is the Stage42-style model4-only custom island. Path B is the Stage43 model4-to-model5 island. Both use the same input, cuts, board ORT runtime, ORT `all`/intra4/inter1/spinning-off contract, CPU0-3, boot, and alternating order. Path B has zero materialized model4-to-model5 transpose and supplies both graph-required suffix inputs.

Before Stage44 repair, A measured `512301.648390 +/- 322.241617 us`; B measured `521603.205250 +/- 148.271914 us`. Paired B-A was `+9301.556860 +/- 451.288699 us`, or `+1.815641%`. Thus Stage43's previously unknown full-island ROI is negative under the measured contract.

The output0 comparison is diagnostic only because the two paths enter the accepted board-ORT divergence surface and discontinuous detection head. Integer model4/model5 boundaries remain governed by fixed-host exact oracles.

Raw evidence: `run_logs/0070_stage44_missing_island_stable.stdout`.
