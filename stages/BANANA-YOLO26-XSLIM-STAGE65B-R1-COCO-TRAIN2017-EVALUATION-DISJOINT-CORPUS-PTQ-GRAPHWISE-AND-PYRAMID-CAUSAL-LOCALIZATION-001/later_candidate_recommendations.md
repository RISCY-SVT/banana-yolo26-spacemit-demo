# Later candidate recommendations

At most two policies are recommended; neither was generated or tested on K1X in this stage.

## 1. Global policy

- Lane: `B2`
- Inference model SHA-256: `40ba6a7f9aebaa98a1c3abe5fce1f66f1bebcd0b10b7af3d26d30414a331d853`
- Host mAP50-95: `0.3658592288412378`
- Host AP-small/medium/large: `0.18014685069413666` / `0.41995524974853976` / `0.5201067045733788`
- Requirement: repeat signed-QDQ conformance and then separately authorize ORT 2.0.6 board placement, correctness, timing, and soak gates.

## 2. Targeted proposal

- Causal classification: `earlier-subgraph-or-tail-interaction`
- Hybrid decision lane: `B2` (selected before full COCO by the frozen scout rule).
- Exact implicated tensor/source-op sites:
  - none selected by the causal gate
- Proposed precision/exclusion: No boundary-specific policy is selected; first localize the earlier inference subgraph or split/tail interaction with additional host evidence.
- Expected accuracy direction: unknown; a pyramid-boundary exclusion is not justified.
- Risk: `precision_level=1` can retain higher-precision regions and may change SpacemiT provider compatibility or fallback. Host graph inventory is not placement evidence.
- Required next gates: deterministic generation, host semantic/COCO repeat, signed-QDQ audit, then separately authorized K1X provider-placement and correctness validation.
