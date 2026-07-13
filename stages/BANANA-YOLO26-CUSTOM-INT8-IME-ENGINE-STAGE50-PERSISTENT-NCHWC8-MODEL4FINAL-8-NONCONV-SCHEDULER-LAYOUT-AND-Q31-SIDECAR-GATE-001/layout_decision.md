# Layout decision

Retain NCHWc8. All four sidecar offset/round-trip contracts were exact for 128/128 checked rows, but no alternative persistent producer-consumer kernel pair was implemented. Therefore no alternative has zero-conversion mean and p95 evidence meeting the 10% promotion rule. The result is `contract-proven-consumer-kernels-not-implemented`, not an alternative-layout performance claim.
