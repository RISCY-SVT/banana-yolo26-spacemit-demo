# Next Blocks Ranked

Stage40 has not yet split the 966-node suffix into fine-grained block timings. Ranking therefore uses current cut-level evidence plus graph adjacency.

| rank | region | evidence | recommendation |
|---:|---|---|---|
| 1 | suffix after `/model.4` | ORT CPU fallback suffix is 129572.132 us and contains 966 nodes | split into block cuts and rank |
| 2 | immediate post-model4 `/model.5/conv/Conv` | first node after the closed `/model.4` cut | include as first expansion candidate |
| 3 | `/model.6/...` block | follows `/model.5` and contains multiple Conv/QDQ/C2f-like operations | profile as a larger expansion candidate |
| 4 | prefix before `/model.4` | prefix cut is 56796.029 us | revisit after suffix ranking |
| 5 | `/model.4` | custom insertion is correctness-closed | do not micro-tune by default |

Next stage should build block-level suffix cuts and choose one expansion target from measured suffix buckets.
