# PMU build report

No installed `perf` binary or authoritative matching kernel source with the SpacemiT X60 pmu-events table was available. Building from an arbitrary source tree was rejected. A stage-local `perf_event_open` wrapper was cross-built instead and run once per basic event under bounded `sudo -n`. No persistent system change was made.
