# Double-buffer pipeline

CPU0-3 execute inference, CPU4 controls the executor, and CPU5-7 prepare the next image. Two explicit buffers transfer ownership once per frame; no IME runs on CPU4-7.
