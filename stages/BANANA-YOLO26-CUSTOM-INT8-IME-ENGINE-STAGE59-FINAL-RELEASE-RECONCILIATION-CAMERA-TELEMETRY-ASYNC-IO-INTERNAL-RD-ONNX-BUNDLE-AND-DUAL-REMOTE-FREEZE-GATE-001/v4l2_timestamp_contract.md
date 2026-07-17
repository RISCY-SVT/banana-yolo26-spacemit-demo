# Direct V4L2 Timestamp Contract

The standalone MMAP probe opens the reviewed `/dev/video20` capture node,
requests an exact format and frame interval, queues four MMAP buffers, then
records every successful `VIDIOC_DQBUF` before immediately requeueing it.

Each row preserves buffer index/count, sequence, kernel timestamp, timestamp
clock and source flags, bytes used, raw flags, local monotonic return time,
poll-wait time, requeue time, and sequence gap. The measured raw V4L2 rate is
computed from first-to-last dequeue return times.

`V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` establishes the timestamp clock.
`V4L2_BUF_FLAG_TSTAMP_SRC_SOE` means start-of-exposure as defined by the driver
ABI; EOF is reported only when the source mask is its EOF value. This is still
not a complete sensor-to-screen chain because OpenCV decode and GUI presentation
do not preserve a correlated kernel buffer timestamp.
