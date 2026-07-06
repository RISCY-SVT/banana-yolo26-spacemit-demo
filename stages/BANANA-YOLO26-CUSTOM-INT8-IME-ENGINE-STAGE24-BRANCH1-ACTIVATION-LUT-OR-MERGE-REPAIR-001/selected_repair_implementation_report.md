# selected_repair_implementation_report

selected_lane: `B`
selected_candidate: `B3_split1_concat_lut_scalar_add`

## Code Changes

The repair is integrated into the real local `/model.4` C2f runner path through an explicit merge mode:

```text
Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT
```

The mode is accepted by both normal model4 C2f runner validation and the ONNX-cut runner validation, but it is not a global/default backend and does not expand the graph.

## Implementation

The repair adds two small per-runner prepared lookup tables:

```text
split1_to_concat_lut_s8[256]
split1_dequant_f32_lut[256]
```

The optimized merge path:

```text
1. copies the already accepted split0 concat LUT output;
2. maps split1 signed storage to concat signed storage through a 256-code LUT;
3. reuses split1 dequantized float values from a 256-code LUT for the Add slot;
4. keeps the Add+post-Concat QDQ scalar exact path for branch1_act_f32 + split1_dequant;
5. preserves the existing final output QuantizeLinear RVV RNE path from Stage23.
```

The failed first attempt is recorded in logs: the candidate initially did not build split0 LUT for the new merge mode in normal runner mode. This was fixed by routing `build_split0_concat_lut_activation` through `merge_mode_uses_split0_concat_lut`.

## Scope

No graph expansion, full engine path, default dispatch, `/data/ncnn` mutation, XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot, CPU4-7 IME, COCO/mAP, camera, or production claim was introduced.
