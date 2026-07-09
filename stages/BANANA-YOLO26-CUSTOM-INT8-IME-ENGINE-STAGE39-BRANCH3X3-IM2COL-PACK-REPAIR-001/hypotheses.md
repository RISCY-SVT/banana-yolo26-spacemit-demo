# Stage39 Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001

H1 fused_im2col_pack_branch0:
  Replacing materialized im2col/gather + pack for /model.4/m.0/cv1/conv/Conv with fused im2col-to-pack or no-intermediate pack reduces branch0 im2col_pack_us.

H2 fused_im2col_pack_branch1:
  Same for /model.4/m.0/cv2/conv/Conv; branch1 has different Cin/Cout and may need a separate path.

H3 interior_fast_path:
  Splitting interior pixels from edge/padding pixels reduces checks/copies and improves pack time without changing output bytes.

H4 row_tile_reuse:
  Reusing row/K-block tiles between neighboring output positions can reduce memory traffic, but only if the overhead does not exceed the savings.

H5 memory_floor:
  If im2col/pack remains bandwidth-bound and no local repair reaches acceptance, stop with a memory-floor report and recommend full-model skeleton / memory-planner work instead of more micro-tuning.
