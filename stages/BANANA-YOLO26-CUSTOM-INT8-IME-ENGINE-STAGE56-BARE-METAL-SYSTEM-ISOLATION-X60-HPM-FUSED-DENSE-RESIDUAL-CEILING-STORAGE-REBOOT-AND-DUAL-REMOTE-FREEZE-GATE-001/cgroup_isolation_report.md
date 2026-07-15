# Cgroup isolation

O1 creates a cgroup-v2 isolated cpuset for CPU0-4 and moves system/user/init slices to CPU5-7. O2 adds movable IRQ, workqueue, and service housekeeping. Effective masks were verified and all state was restored after each arm.
