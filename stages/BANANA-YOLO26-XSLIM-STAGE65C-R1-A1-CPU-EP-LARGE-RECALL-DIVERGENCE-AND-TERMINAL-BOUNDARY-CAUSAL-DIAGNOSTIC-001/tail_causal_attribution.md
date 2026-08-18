# Common-tail causal attribution

## Contract

- Frozen B2 and A1 CPU/SpacemiT sessions emitted the same ordered six-boundary contract.
- Every captured boundary set was replayed through the exact common float tail with SHA-256
  `18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3`.
- Board replay, host replay, and repeated host replay reproduced each original surface exactly.
- Single-boundary splices replace one EP boundary with its same-model CPU boundary; model bytes,
  the other five boundaries, and the tail remain frozen.

## Findings

Across all 64 preselected cases, confidence outputs carry the largest A1-specific normalized
CPU/EP interaction: P3 confidence is 1.456 exported-qparam steps RMS, P5 confidence 1.407, and
P4 confidence 1.284. The corresponding bbox values are 0.568, 0.536, and 0.519 steps.

For the 32 deterministically selected large-recall-loss cases, the first single boundary that
recovers the largest part of the A1 EP-to-CPU tail-output distance is P5 confidence: mean
recovery 0.048313. P3 confidence recovers 0.033949 and P4 confidence 0.025413. Each bbox splice
recovers less than 0.0001. The result is therefore confidence-domain specific, with P5 the
strongest large-loss boundary, but it is distributed across all three confidence scales rather
than explained by one boundary alone.

The common tail materially changes ranks and TopK membership. On the large-loss subset, a
single confidence splice moves the final output measurably toward CPU while bbox splices are
effectively inert. Threshold-crossing and TopK tables show that small terminal confidence
differences can change score ordering and membership after the tail. The divergence is already
present at the terminal confidence boundaries; the discontinuous tail amplifies it into decoded
rank and recall changes.

## Narrow conclusion

`terminal-confidence-difference-with-tail-rank-amplification` is supported for the selected
diagnostic cases. P5 confidence is the first material single-boundary explanation for the
large-loss subset. This diagnostic does not by itself establish a population-level provider
interaction; the shared-draw full-val difference-in-differences interval remains the authority.
