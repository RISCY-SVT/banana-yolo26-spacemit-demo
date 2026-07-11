# Model5 correctness

Correctness authority is independent integer/operator semantics plus fixed host ORT 1.27.0 CPU EP under `ORT_DISABLE_ALL`. Board ORT is diagnostic/timing only.

Final R2a results:

- Host scalar F0: model4 postactivation and model5 output mismatches 0, max abs diff 0.
- Board IME F0-F7: mismatches 0, max abs diff 0 at model4 postactivation and model5 output.
- Board F0 with 1/2/3/4 workers: exact for every arm, `affinity_ok=1`.
- FRM RNE/RTZ/RDN/RUP/RMM: output exact and ambient `frm` restored in every case.
- Repeated F0 hashes were stable; no SIGILL; no CPU4-7 IME execution.

Final F0 model5 SHA-256 is `c65bc733396dc788bf3ee9861f3c9a6fbffc6793009a07b2ae324eb61644a1b2`. The complete fixture hashes are in `model5_correctness_matrix.tsv`.
