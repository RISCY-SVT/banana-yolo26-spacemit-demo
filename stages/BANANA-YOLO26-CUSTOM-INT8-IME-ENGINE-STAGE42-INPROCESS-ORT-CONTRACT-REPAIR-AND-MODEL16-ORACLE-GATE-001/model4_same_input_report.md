# Model4 Same-Input Report

## Fixed input

- tensor: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`
- dtype/layout/shape: uint8, NCHW, `1x64x80x80`
- input NPY SHA-256: `a2f9fa064d0e828efd6cd6f389cf69b9aa40b1461ead61188dffce2399734262`
- input raw SHA-256: `ae3ee976dad8c640c6357d43c4810c8fd07103e1627ced0ae4d125446bd485d4`
- cut SHA-256: `bde82b0130615717ffcbdbaca8fa274e5de00c111cf0b0a518023b6a674d841a`
- fixed expected output raw SHA-256: `b3e3410a9e7476ef01c3e65a2b6cddc6ab97e6e930a9dace544769385c515d2e`

Every arm consumed these identical input bytes. No prefix-derived tensor was substituted.

## Operational `ORT_ENABLE_ALL` matrix

| Comparison | Mismatches | Max abs diff | Result |
|---|---:|---:|---|
| host ORT vs board ORT | 77196 / 819200 | 2 | mismatch |
| host ORT vs host scalar | 0 | 0 | exact |
| host scalar vs board scalar | 0 | 0 | exact |
| board scalar vs board IME | 0 | 0 | exact |
| host ORT vs board scalar | 0 | 0 | exact |
| host ORT vs board IME | 0 | 0 | exact |
| board ORT vs board scalar | 77196 / 819200 | 2 | mismatch |
| board ORT vs board IME | 77196 / 819200 | 2 | mismatch |

The host ORT, host scalar, board scalar, and board IME outputs share raw SHA-256 `b3e3410a...15d2e`. Board ORT output SHA-256 is `35554031...cfb09`.

## `ORT_DISABLE_ALL` diagnostic

- host ORT vs fixed output: 1 mismatch, max diff 1.
- board ORT vs fixed output: 3 mismatches, max diff 1.
- host ORT-disable vs board ORT-disable: 4 mismatches, max diff 1.
- host scalar-disable vs board scalar-disable: exact.

This proves that graph optimization level is part of the accepted export contract while also showing that disabling optimization does not fully match host and board ORT.

## Interpretation

There is no Stage42 scalar or IME correctness blocker. The prior Stage41 comparison used board ORT as authority and therefore attributed the board runtime's differing output to the custom implementation. With one fixed input, the custom scalar and IME routes are byte-exact against the fixed host oracle.

Complete statistics are in `model4_same_input_matrix.tsv`; provenance is in `model4_same_input_dumps_manifest.tsv`.
