# D8 causal decision

Classification: `upstream-branch-error-material`.

- D8 model SHA-256: `a77f2efea1dee7578d66859159a01c08ea45b76b44865d40b813732aa84372d4`.
- H500 recovery fraction: `0.335397387`.
- Full-val recovery fraction: `0.374970551`.
- Full-val H8-D8 gap: `0.022477663` mAP50-95.
- Full-val D8-H0 bootstrap 95% interval:
  `0.013419620989654905` to
  `0.017029646442466354`.
- Full-val D8-H8 bootstrap 95% interval:
  `-0.02467885166774615` to
  `-0.019541507677064834`.

D8 removes only the six final output Q/DQ pairs. It recovers a significant but
minority fraction of the H0-to-H8 gap; accumulated upstream branch error is
therefore material. D8 is host-diagnostic-only and is not deployable/provider
evidence.
