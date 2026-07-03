# Static Model Format v0

## Goals

The runtime is model-specific and static. It must not depend on ONNX Runtime,
protobuf, Python, a dynamic graph executor, or ncnn.

## Header

```text
magic: Y26I8K1X
version: 0
endianness: 0x01020304
alignment: 64 bytes preferred
header_size: sizeof(ModelHeaderV0)
tensor_table_offset
op_table_offset
weight_blob_offset
scale_blob_offset
string_table_offset
checksum
model_contract_id
quantization_profile_id
```

`custom_int8_engine/include/y26_k1x_engine.h` contains the Stage 0 C++ skeleton
for `ModelHeaderV0` and `ScaleDescriptorV0`.

## Tables

Tensor table minimum fields:

```text
name_offset
dtype
rank
shape_offset
scale_descriptor_index
zero_point
layout
blob_offset
```

Op table minimum fields:

```text
op_type
input_tensor_range
output_tensor_range
attribute_offset
kernel_profile_id
```

Scale descriptor:

```text
dtype: fp32 or fixed-point
granularity: per-tensor/per-output-channel/per-group
axis
count
alignment
blob offset
```

## Checksums

The v0 checksum covers all bytes after setting the checksum field to zero. The
exact checksum algorithm is deferred until the converter exists; Stage 0 uses
the field but does not produce a full model binary.

## Contract IDs

Initial IDs:

```text
model_contract_id 1: yolo26n_640_e2e_1x300x6
model_contract_id 2: yolo26n_640_traditional_1x84x8400
quantization_profile_id 1: manual_ort_qdq_conv_matmul_representative
```
