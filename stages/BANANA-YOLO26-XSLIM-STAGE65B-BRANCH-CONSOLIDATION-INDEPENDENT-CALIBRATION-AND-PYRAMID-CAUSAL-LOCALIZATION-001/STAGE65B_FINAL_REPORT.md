# Stage65B final report

## Classification

`stage65b-blocked-independent-calibration-corpus-missing`

Publication classification: `stage65b-consolidation-published-host-accuracy-not-executed`.

## Outcome

The one-time XSlim branch consolidation completed without rewriting history.
The released `2.1.2+riscy.1` wheel was installed by immutable hash in a fresh
stage-local environment. Gate 3 then stopped the host accuracy study: no
licensed, independent calibration corpus large enough for the mandatory
50/200/500/1000 matrix exists under the inspected `/data` roots.

The only large calibration tree contains 2,015 images. Content hashing proves
that all 2,015 are byte-identical members of COCO val2017, leaving zero
independent images. The project fixture trees contain only 45 unique images;
32 overlap val2017 and the remaining 13 are private/synthetic/derived fixtures,
not a licensed independent detection corpus and not enough for C50.

Per the explicit stop condition, no Stage65B PTQ, preprocessing qualification,
semantic validation, Graphwise run, boundary audit, hybrid ablation, or COCO
evaluation was started. No Stage65C candidate was selected.

## Completed gates

- Protected Banana, XSlim release, custom-executor, and `/data/ncnn` identities matched.
- Stage64 and Stage65A-PUB4 evidence identities were imported and verified.
- `riscy/k1x-yolo26` was created from the immutable release commit with one policy-only commit.
- Evidence tag `evidence/stage65a-pub4-closure-001` preserves the PUB4 evidence history.
- The new branch and evidence tag have GitHub/GitLab parity and protection.
- Exactly five redundant remote RISCY branches were deleted after reachability proof.
- Release tag `v2.1.2-riscy.1`, main, release assets, Actions-disabled state, and PyPI absence remain unchanged.
- The released wheel SHA-256 is `635441d26458c6754627dd9595132cfd9a16d762d3ed0252f0039be668c01784`.

## Corpus proof

| Surface | Rows | Unique | val2017 overlap | Independent qualifying images |
|---|---:|---:|---:|---:|
| COCO val2017 | 5,000 | 5,000 | 5,000 | 0 by definition |
| Existing `coco_calib2K` | 2,015 | 2,015 | 2,015 | 0 |
| Project fixture collection | 109 | 45 | 32 unique | 13 non-corpus fixtures |

The full overlap TSV has SHA-256
`6a5c0873bd60db72070999ce49eb05123af57814b50e767f919f39ae91a90a75`.
The independent residual is empty and hashes to the standard empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Not executed

Gates 4 through 10 are recorded as
`not-run-blocked-independent-calibration-corpus-missing`. Stage64 values in
the comparison files are imported controls only, not new Stage65B results.
No board command, runtime promotion review, source patch, release mutation,
training, QAT, targeted model generation, or issue update occurred.

## Required human input

Provide an existing, licensed detection corpus with at least 1,000 distinct
images, documented provenance, and zero content-hash overlap with COCO
val2017. Annotations are additionally needed to construct the requested
size-balanced C500 lane. A fresh explicit authorization should then resume the
host accuracy study; Stage65C is not ready.
