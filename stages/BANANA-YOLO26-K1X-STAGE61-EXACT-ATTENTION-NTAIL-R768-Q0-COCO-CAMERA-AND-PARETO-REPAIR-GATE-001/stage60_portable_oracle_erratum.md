# Stage60 Portable-Oracle Erratum

Stage61 found an evidence defect in the accepted Stage60 full-fixture parity
surface. The K1X board's RVV input path multiplies each float input by 255 in
float32 and then applies explicit RNE. The host fallback widened the input to
double before multiplication. These operations differ at exact tie-adjacent
values in the generated F6 vertical-ramp and F7 random fixtures.

The Stage60 `fixture-parity.final.tsv` table reported `boundary_count=0` and
`pass` even though its host and board manifests differ for those fixtures.
That aggregate wording was therefore too broad. Stage60 board scalar versus
board optimized equality, real-image outputs, package identities, COCO, and
performance evidence remain valid.

Stage61 corrects only the non-RVV portable fallback so that it emulates the
already-frozen RVV float32-multiply/RNE contract. The K1X implementation,
qparams, integer operators, package data, and selected output bytes are not
changed. All host oracles are regenerated after this correction and compared
against fresh board scalar and optimized dumps rather than inherited pass
labels.
