# Asynchronous Recording Decision

**Selected:** bounded asynchronous MJPG recording with one CPU6 owner thread,
queue depth two, and replace-old behavior.

On the matched 640x480 MJPG full-GUI surface, no recording processed 6.578935
FPS. Synchronous recording fell to 6.077503 FPS (-7.621781%). Asynchronous
recording processed 6.562994 FPS (-0.242304%), within the 2% selection gate.
It wrote 1,212 readable MJPG frames over 178.858 seconds with zero queue
replacements and zero write failures. Sequence/enqueue/write metadata remained
paired in `async-record.avi.frames.tsv`.

Shutdown under SIGINT, SIGTERM, and SIGHUP produced readable AVIs and complete
metrics. The AVI uses the measured initial processing cadence as its nominal
rate; `*.frames.tsv` is authoritative for variable-rate source sequence and
write timing.
