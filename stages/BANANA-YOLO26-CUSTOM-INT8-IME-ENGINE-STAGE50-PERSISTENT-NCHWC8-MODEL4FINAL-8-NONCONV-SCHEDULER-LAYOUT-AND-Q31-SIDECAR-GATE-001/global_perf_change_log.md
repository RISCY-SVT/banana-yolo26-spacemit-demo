# Global perf configuration

The board began with `kernel.perf_event_paranoid=2` and `kernel.kptr_restrict=1`. Stage-owned bounded commands temporarily set them to `-1` and `0`, then restored `2` and `1`. `/etc/sysctl.d/99-k1x-perf-lab.conf` was verified absent. No persistent sysctl or file capability was installed; eMMC exception count is zero.
