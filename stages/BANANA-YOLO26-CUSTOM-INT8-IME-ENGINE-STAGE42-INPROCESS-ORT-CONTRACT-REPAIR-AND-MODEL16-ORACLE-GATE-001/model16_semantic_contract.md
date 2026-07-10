# Model16 Semantic Contract

The semantic block contract was verified from graph producers, consumers, inferred value info, and live host ORT outputs.

```text
input name: /model.15/Concat_output_0_DequantizeLinear_Output
input dtype: float32
input shape/layout: 1x256x80x80 NCHW
output name: /model.16/cv2/act/Mul_output_0_DequantizeLinear_Output
output dtype: float32
output shape/layout: 1x64x80x80 NCHW
cut SHA256: a789d7db60056cdd24f6854815ec502a11efc0cedbb1276c3be706e6100136d7
```

The fixed host input NPY SHA-256 is `042903ef08c674bdd7981b3141ca2c686c546d7878c7bed94ad0bb8f18ecb97b`; output NPY SHA-256 is `baa4543817feb9790339215dba9ab115a58f04c501b546b844c2af1bad00a276`.

This surface preserves ONNX semantic boundaries but incurs float materialization. It is an oracle/debug contract, not the preferred custom implementation boundary.
