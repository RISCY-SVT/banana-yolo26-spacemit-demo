# Runtime OS selection

Select O2: CPU0-4 isolated cgroup plus movable IRQ/workqueue/service housekeeping on CPU5-7. Against O0, mean improved 0.792% and max improved 10.28%; three independent selected-source launches averaged 142148.247 us. O3 added no gain and FIFO regressed.
