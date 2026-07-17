# Direct V4L2 Decision

The standalone MMAP sidecar established the raw compressed-buffer surface
without changing the public demo backend. Both MJPG modes dequeued at about
30.002 buffers/s with zero driver-visible sequence gaps, while OpenCV delivered
about 10 decoded frames/s at 1280x720 and 15 decoded frames/s at 640x480 during
matched full-GUI runs.

A direct latest-compressed-buffer decoder was optional. It was not implemented
or promoted because Stage59 did not produce the required full-camera decoded
path A/B, reconnect proof, and 30-minute stability result. The frozen public
route remains OpenCV V4L2. The measured gap is retained as evidence for a
separate camera-backend maintenance task, not reported as achieved throughput.
