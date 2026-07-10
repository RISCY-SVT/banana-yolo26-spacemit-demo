# Stage41 Reproduction Report

## Fixed artifacts

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- input NPY SHA-256: `46908132cf9e04ee14d1702ce99f583255b3a0de0a9b3e74d0f277ab9d8e09b7`
- accepted output0 NPY SHA-256: `d07f34ed645101cf735dd82ea10b9488f8abdc847d431d108ec78154eb238fe7`
- host runtime: ORT `1.27.0`, CPU default provider, API 21.
- reproducing session: optimization `all`, sequential, intra/inter `1/1`, memory pattern and CPU arena enabled, spinning enabled.

## Host reproduction

The repaired C++ runner reproduced all accepted Stage41 host gates:

| Gate | Mismatches | Max abs diff | Result |
|---|---:|---:|---|
| full ORT vs accepted output0 | 0 | 0 | pass |
| custom scalar model4 vs host ORT model4 | 0 | 0 | pass |
| custom model4 through suffix vs full host ORT | 0 | 0 | pass |

The raw host output0 SHA-256 is `8ddc0e17ab7307ac7fc1f91d9145acf3f88647d7528e73183b8e6d723c41ebac`. The model4 output NCHW SHA-256 is `b3e3410a9e7476ef01c3e65a2b6cddc6ab97e6e930a9dace544769385c515d2e`.

`ORT_DISABLE_ALL` is a diagnostic contract, not the accepted export contract: it differs from the accepted output0 in 1509/1800 floats and from the fixed same-input model4 oracle in one uint8 element. Graph optimization policy is therefore part of the reproducibility contract.

## Tensor contract correction

Live graph metadata confirms:

- model4 input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`, uint8, NCHW `1x64x80x80`, custom NHWC `1x80x80x64`.
- model4 output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`, uint8, NCHW `1x128x80x80`, custom NHWC `1x80x80x128`.

The graph-derived manifest is `boundary_tensor_manifest.tsv`.

## Attempt history

Stage42 attempt 1 was only an administrative approval check. It performed no repository, build, board, source, packet, commit, or push work and is superseded by this direct-user-authorized technical rerun.
