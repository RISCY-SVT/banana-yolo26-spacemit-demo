# Prefix implementation

The persistent prefix covers the frozen input quantization surface through `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output` and joins the accepted resident core without a per-operator layout adapter. Preprocessed float input is quantized directly into the resident `NCHWc8_SPATIAL_INNER_V1` arena. The RGB API writes already-letterboxed RGB bytes directly into the same resident tensor.

Dense Conv uses the exact M12xN16 IME route and Q62 E2c where eligible. The 8-channel model2 path currently uses the exact masked N16 functional route; a true N8 performance kernel remains a documented optimization item. There is no ORT, Python, float Q/DQ materialization, or per-run allocation in the measured prefix.
