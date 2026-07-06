# Fullshape Oracle Reproduction Commands

```bash
cd /data/banana-yolo26-spacemit-demo
.deps/custom_int8_engine/venv-stage20-onnx/bin/python \
  custom_int8_engine/tools/extract_fullshape_oracle.py \
  --model .deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx \
  --out-dir .deps/custom_int8_engine/stage20_fullshape_oracles/model4_c2f_synthetic_seeded \
  --input-mode synthetic_seeded \
  --boundary-set model4_c2f \
  --manifest stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001/fullshape_boundary_manifest.tsv \
  --checksums stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001/fullshape_tensor_checksums.tsv \
  --metadata-json stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001/fullshape_oracle_metadata.json
```

The generated `.npy` files are large and stay outside git under `.deps/`.
