# XSLIM-DEV-001A final report

## Decision

- Stage: `BANANA-YOLO26-XSLIM-DEV-001A-SPACEMIT-S8-QDQ-CONSTRAINED-RANGE-OBSERVER-TERMINAL-DOMAIN-AND-POLICY-A-HOST-CANDIDATE-GATE-001`
- Classification: `xslim-dev-001a-all-s8-policy-a-host-pass-candidate-freeze-ready-for-separate-k1x-gate`
- Publication classification: `research-development-only-not-published`
- Full-val winners: `A1`
- Positive below-gate candidates: `none`

## Source and package

XSlim development version `2.1.2+riscy.2.dev1` adds deterministic strict local
selection, constrained signed asymmetric INT8 range search, frozen qparam manifests,
and the structural `spacemit_k1x_s8_qdq_split_v1` validator. No-override B2 generation
is byte-identical to the frozen B2 deployable/inference/tail artifacts.

| artifact | sha256 | status |
| --- | --- | --- |
| xslim-2.1.2+riscy.2.dev1-py3-none-any.whl | c31afd3a0f1479e55e242d162b25a203e4511e7ea6a8c3e71eb3232dc92de6b8 | byte-identical-across-two-clean-builds |
| xslim-2.1.2+riscy.2.dev1.tar.gz | d8d14ffa920ded5b7befe030f85fbbef3ac7a371f66c69ef0bfafe673cefd9b6 | normalized-byte-identical-across-two-clean-builds |

Full tests: 174 passed, 2 inherited warnings, 65 subtests. Focused contracts: 27 passed.

## Candidate contract

All A1-A6 candidates reproduced byte-for-byte across two clean generations and passed:
812 Q/DQ nodes, 0 QLinear, 0 UINT8 zero points, 0 FP16, 102/102 explicit Conv
`kernel_shape`, exact six-output order, exact FP32 tail, profile validation, fixtures,
and 100-image semantic/collapse checks.

## H500

| surface | map50_95 | ap_small | ap_medium | ap_large | prediction_sha256 |
| --- | --- | --- | --- | --- | --- |
| A0 | 0.4446654879525213 | 0.24198226769512168 | 0.5163766839660254 | 0.6768001039275425 | 060928fd2c30e62b5cc533747c53b0c9157c5483d8c77827bbdcbec828db5ae9 |
| A1 | 0.4517284340421088 | 0.23861981266752952 | 0.5215365636326764 | 0.6887924786161271 | aae66e9e98c2ee4b10aff3385a5057da834680ff0ac8aba9e93a0baf56a4b4bf |
| A2 | 0.43828401678858836 | 0.2249152683173927 | 0.5093272053797157 | 0.67984859514682 | 1736e4274f9e282fe550e2a0ac9132a675bed20c4e8445fb8acd351c9445d12b |
| A3 | 0.44735060862971726 | 0.23318824973911687 | 0.5135221920312817 | 0.6819178477843859 | 57b2e2428a1e66fa738d321d5654936f7060b892b1d2850c60f5ce74eebc44d3 |
| A4 | 0.4470981709935015 | 0.23221143191424265 | 0.5124130551156666 | 0.6811823147763165 | 149f9d221b1c90a8cb49186dbbb0b1a3d466dda9cc2d573749e1b225154fd02f |
| A5 | 0.4504568584671628 | 0.22828817977955493 | 0.5158261864289463 | 0.6990811396836467 | 4140d553f3000804c59a061fd580fd0d5766710bd21f092dedeb3ea5fa5d49d0 |
| A6 | 0.4519834481892158 | 0.22928913717241906 | 0.5166533337707343 | 0.7010458851467165 | 907c3ef4f6bc2541f92888c92818a9a893b4b0972384447e4d4620c698599139 |

The only H500-qualified lane was A1 (T6 terminal ranges): +0.007062946 mAP versus B2;
10,000-replicate 95% CI 0.002017412..0.013001502. It recovered 23.24% of the accepted
B2-to-FQ8-L3 H500 oracle gap, below the 50% strong-success label.

## Full val2017

| surface | map50_95 | ap_small | ap_medium | ap_large | prediction_sha256 |
| --- | --- | --- | --- | --- | --- |
| A0 | 0.3658592288412378 | 0.18014685069413666 | 0.41995524974853976 | 0.5201067045733788 | 51f8d4b25245a5f3e24feafea8aa49547c0f530f59cabcd18e61a744b4740add |
| A1 | 0.37293515356510193 | 0.1798466891332123 | 0.424335191892151 | 0.5412885140154299 | fdae3c397ff82b005b3c0f507496392dde381fed2aaa0f5d18f03ea35c7b2df9 |

The final decision is taken from `full_val_candidate_decision.tsv` and its 10,000
paired image-level COCO bootstrap. FQ8-L3 and H8 are diagnostic ceilings, not deployable
candidate claims.

## Scope

No K1X board command, provider-placement test, performance run, Policy B implementation,
custom-executor mutation, release publication, or tag mutation occurred.
