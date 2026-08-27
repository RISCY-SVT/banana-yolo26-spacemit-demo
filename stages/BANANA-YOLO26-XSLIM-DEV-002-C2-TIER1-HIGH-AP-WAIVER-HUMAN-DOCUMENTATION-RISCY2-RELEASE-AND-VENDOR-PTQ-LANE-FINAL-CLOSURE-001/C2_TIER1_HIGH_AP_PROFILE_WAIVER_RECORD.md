# C2 TIER-1 High-AP Profile Waiver

Record ID: `BANANA-YOLO26-C2-HIGH-AP-PROFILE-TIER1-WAIVER-001`

This is an append-only human disposition. It does not alter the historical
Stage65D-R1 classification or its universal non-inferiority gate.

## Decision

- Historical C2 universal gate: **FAIL, unchanged**.
- B2 remains the universal vendor control and rollback.
- C2 is approved only as a separate frozen higher-AP application/research
  profile under this TIER-1 waiver.

The approved bytes are bound by:

| Artifact | SHA-256 |
|---|---|
| C2 deployable | `e963be11c57c048f23caa34df1e2d140211632cc4dfd6b734b14909a30ea4b55` |
| C2 inference | `281f4acd1261e7ee2c38b6e3bdecbf61c3d91cf710c63e6bc6cdaf257a52669b` |
| Common float tail | `18ffff41e6812fa781baf7b9c1fcd41b41d6118145d785c3e550499070a512a3` |

## Accepted Evidence

Board SpaceMIT EP, C2 minus B2:

- mAP50-95: `+0.012599140370`
- AP-large: `+0.039947408575`
- AR-small: `-0.003444784993`
- AR-large: `-0.003078277927`

Matched performance was equivalent within the empirical noise floor. B2 and
C2 each passed 10,000-run stability.

## Operating-Point Warning

At score `0.25`, IoU `0.50`, and `maxDets=100`, C2 has fewer false
positives but more false negatives than B2. Stage65E proves that this trade-off
depends on score threshold.

A C2-specific score threshold must therefore be selected and validated against
the application's false-negative and false-positive costs before C2 can become
an application default. The threshold must not be inherited silently from B2.

## Limits

This waiver does not authorize:

- replacement of B2 as the universal baseline;
- runtime default promotion;
- model, dataset, or vendor-binary publication;
- camera or field-performance claims;
- model or qparam mutation;
- another quantization campaign.

Any application using C2 must retain B2 as rollback and record the selected
threshold, target classes, operating environment, and acceptance owner.

Recorded by direct human authorization in DEV-002.
