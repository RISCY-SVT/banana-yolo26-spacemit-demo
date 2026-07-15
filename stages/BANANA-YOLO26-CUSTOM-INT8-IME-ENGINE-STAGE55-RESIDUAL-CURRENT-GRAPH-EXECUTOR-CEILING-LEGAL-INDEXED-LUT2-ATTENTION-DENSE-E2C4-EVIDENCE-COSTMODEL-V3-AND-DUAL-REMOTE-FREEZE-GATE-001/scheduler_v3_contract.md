# Scheduler V3 contract

`begin_active_window()` wakes persistent workers for one inference; `end_active_window()` makes them park on a condition variable. Workers retain the exact epoch-spin dispatch while active and never spin across camera-like inter-frame gaps. SCHED_OTHER remains mandatory.
