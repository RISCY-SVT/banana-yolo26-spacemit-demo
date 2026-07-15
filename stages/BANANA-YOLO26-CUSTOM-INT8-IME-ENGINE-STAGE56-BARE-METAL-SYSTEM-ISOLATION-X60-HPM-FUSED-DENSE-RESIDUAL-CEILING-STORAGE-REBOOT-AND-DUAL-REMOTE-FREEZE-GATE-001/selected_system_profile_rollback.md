# Selected profile rollback

Run `scripts/stage56-system-profile.sh restore <state-dir>`. It restores IRQ masks, workqueue mask, systemd AllowedCPUs, service states, cgroup partition state, and removes the active marker. Rebooting the unchanged B0 entry is the final recovery path.
