# Rollback plan

Runtime O2 snapshots IRQ masks, workqueue mask, systemd AllowedCPUs, services, and cgroup state before apply. Restore is idempotent and runs on normal exit and signals. The original boot entry remains the only selected boot profile; no kernel, DTB, or command line changed.
