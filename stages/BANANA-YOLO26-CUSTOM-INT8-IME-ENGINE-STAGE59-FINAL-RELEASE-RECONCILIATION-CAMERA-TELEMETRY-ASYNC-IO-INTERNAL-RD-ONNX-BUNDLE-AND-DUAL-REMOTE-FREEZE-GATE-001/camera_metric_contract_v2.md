# Camera Metric Contract V2

`metrics_schema_version=2` removes the ambiguous Stage58 names.

| Field | Exact meaning |
|---|---|
| `requested_fps` | Value requested from the capture backend. |
| `backend_reported_fps` | Value returned by OpenCV after configuration. |
| `opencv_decoded_frame_fps` | Frames for which OpenCV `read()` returned a decoded image during the measured window. |
| `captured_total` | Decoded images returned since the capture thread started, sampled at measured-window end. |
| `captured_measured` | Decoded images returned after the warmup snapshot and before measured-window end. |
| `application_slot_replacements_*` | A decoded frame replaced an older unconsumed frame in the depth-one application slot. |
| `processed_fps` | Completed inference/render/display calls per measured-window second. |
| `displayed_fps` | Frames submitted to the GUI display call per measured-window second. |
| `wait_for_slot_ms` | Consumer wait from loop entry until a decoded frame is available. |
| `consumer_loop_ms` | Decode-return consumer work through display/event handling for that frame. |
| `decoded_read_return_to_display_call_ms` | Time from OpenCV `read()` return for that frame through its display call. |
| `shutdown_finalize_ms` | Post-window capture join, recorder drain, evidence flush, and final-output time. Excluded from processed FPS. |

None of these fields is raw sensor FPS or sensor-to-screen latency. Direct V4L2
sequence and timestamp evidence is reported separately.

Producer counters are snapshotted at warmup completion. Summary FPS uses an
explicit measured-window end captured before thread joins, recorder drain, or
file flush.
