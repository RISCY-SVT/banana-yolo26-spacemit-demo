# DEV-001C full-val decision

Decision: `fail`.

Disposition: full-val Pareto fail; vendor PTQ lane closed.

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| C2-B2 mAP point | 0.01264954912663352 | >=0.005 | pass |
| C2-B2 mAP CI lower | 0.011938434367132782 | >0 | pass |
| C2-B2 ap_small point | 0.00031421039660825545 | >=-0.003 | pass |
| C2-B2 ap_small CI lower | -0.0010998736510771016 | >=-0.005 | pass |
| C2-B2 ap_medium point | 0.005676570355646338 | >=-0.003 | pass |
| C2-B2 ap_medium CI lower | 0.004964381626665719 | >=-0.005 | pass |
| C2-B2 ap_large point | 0.038616455035753505 | >=-0.003 | pass |
| C2-B2 ap_large CI lower | 0.03142742272864314 | >=-0.005 | pass |
| C2-B2 ar_small point | 0.000137312257035338 | >=-0.003 | pass |
| C2-B2 ar_small CI lower | -0.0010784950575415662 | >=-0.005 | pass |
| C2-B2 ar_medium point | -0.00030788462241371217 | >=-0.003 | pass |
| C2-B2 ar_medium CI lower | -0.0022344874731478393 | >=-0.005 | pass |
| C2-B2 ar_large point | -0.0024025049065128368 | >=-0.003 | pass |
| C2-B2 ar_large CI lower | -0.004377891003094861 | >=-0.005 | pass |
| C2 mAP Pareto vs A1 | 0.005573624402769373 | >=-0.001 | pass |
| C2 AR-large repair vs A1 | 0.0011201782479843825 | >=0.002 | fail |
| P(C2-A1 AR-large>0) | 0.876 | >=0.95 | fail |
