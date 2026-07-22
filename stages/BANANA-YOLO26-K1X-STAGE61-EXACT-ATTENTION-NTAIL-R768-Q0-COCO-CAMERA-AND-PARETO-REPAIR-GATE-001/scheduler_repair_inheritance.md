# Scheduler Repair Inheritance

The Stage60 research head already contains the two liveness repairs later
published in release 0.9.3:

- threaded-convolution readiness is published under the mutex used by the
  creator's condition-variable predicate;
- active-window transitions are serialized under the lifecycle mutex, workers
  recheck state before parking, park/wake acknowledgement is counted, stale
  generations are rejected, and destruction wakes parked workers.

Stage61 did not reimplement or alter those repairs. Native, ASan/UBSan and TSan
CTest each passed 54/54 tests. Twenty repeated startup runs covered 8,000 worker
workspaces and 20,000 readiness publications. Twenty repeated active-window
runs covered 40,000 transitions. A board 2,000-transition smoke also passed.
No TSan race report, timeout, stale replay, or unsupported partial-worker mode
was observed. Stage61 uses all workers complete with the pool at capacity.
