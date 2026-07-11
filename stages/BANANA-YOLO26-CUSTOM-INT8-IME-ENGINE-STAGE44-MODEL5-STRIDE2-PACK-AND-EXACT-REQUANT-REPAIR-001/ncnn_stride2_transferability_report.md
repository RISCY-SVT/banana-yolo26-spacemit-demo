# ncnn stride-2 transferability

Read-only source: `/data/ncnn`, commit `684074c67d9a33fd82b2c630b062dd822dea22c8`. No `/data/ncnn` file was modified.

The accepted ncnn route uses planar pack1 input, `vlseg2`/`vlse8` stride-2 loads, two-output-channel A reuse, OC-parallel OpenMP, and float output. Its guard requires exact 3x3s2, no int8 requant, no activation, even outputs, and a packed float output layout. Its accepted reduction was route-specific `13.016%`.

YOLO26 model5 uses NHWC signed-code storage, small MMT4D A panels, prepacked B, persistent spatial-row workers, explicit zero-point correction, per-channel fixed integer requant, and SiLU LUT output. The ncnn code is therefore not a drop-in patch. Only the direct stride-2 gather and A-reuse concepts transfer algorithmically.

Stage44 used that bounded concept in R2a by copying channel chunks directly into the existing four-position panel. It did not copy ncnn source or broaden the accepted ncnn dispatch.
