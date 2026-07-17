# Overlay Semantics

The current frame displays current capture, preprocessing, inference, and
postprocessing values. Rendering, display-call, and complete consumer-loop
values cannot be known until after that frame's overlay is drawn, so the panel
labels those fields explicitly as `prev_render`, `prev_display`, and
`prev_consumer`.

The FPS labels are `processed_fps` and `decoded_fps`; queue loss is labeled
`slot_replaced`. Requested/backend-reported camera mode remains in the camera
line and is not presented as a measured sensor rate.
