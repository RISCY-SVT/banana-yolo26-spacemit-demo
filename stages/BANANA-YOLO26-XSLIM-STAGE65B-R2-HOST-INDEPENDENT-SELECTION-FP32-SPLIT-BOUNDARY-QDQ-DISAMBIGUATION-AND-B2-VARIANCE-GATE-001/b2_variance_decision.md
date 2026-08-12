# B2 variance decision

Decision: `no-significant-aggregate-map-sensitivity-proven`.

H500, not scout500, is the selection and robustness surface.

- `Vseed`: delta versus frozen B2 `+0.006168063` mAP50-95; 95% CI `-0.001193573` to `+0.013924200`; P(delta>0) `0.94`; model `81ae8ac31aca734c3b0ca3808476d7290ae25588622298eb1e26abdae8c872e3`.
- `Vorder`: delta versus frozen B2 `-0.000409460` mAP50-95; 95% CI `-0.006726854` to `+0.007143565`; P(delta>0) `0.596`; model `5ab49bff35f60c1927c6c72551463a38cd972cc6afe445b3e672a3ce1b63cb40`.
- `Vdraw`: delta versus frozen B2 `-0.001039333` mAP50-95; 95% CI `-0.010463383` to `+0.007430665`; P(delta>0) `0.421`; model `fdbf22f4c6325b17aacc1a57ae790a4028de74352b7d29572f0584b5e0891d72`.
- No >=0.005 aggregate-mAP arm had a two-sided 95% interval excluding zero.
- Membership nevertheless has a size-bin sensitivity signal: Vdraw AP-small delta `-0.020109849` (95% CI `-0.029631594` to `-0.004098977`), and AP-medium delta `-0.031549428` (95% CI `-0.044109010` to `-0.011730076`).
- No variance full-val run was opened. Vseed crossed the +0.005 H500 point gate but P(delta>0) was `0.94`, below the predeclared 0.95 requirement; Vorder and Vdraw did not cross the point gate.
