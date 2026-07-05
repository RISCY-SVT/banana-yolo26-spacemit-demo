# View Based Concat Report

`A4_view_based_concat_packA_from_channel_spans` was not implemented in Stage 13.

Reason:

- The current `/model.2/cv2/conv/Conv` IME API consumes NHWC signed int8 and
  packs A internally.
- A2 already writes exact post-Concat signed int8 NHWC storage directly.
- True multi-span packA would require a narrower Conv packer extension and is
  better evaluated when another block exposes a repeated Concat/pack pattern.

This remains a future candidate, not an accepted Stage 13 path.
