# Scheduler Semantic Reconciliation

The maintenance and Stage61 scheduler implementation files are byte-identical. The merged protocol preserves readiness publication under the predicate mutex, active-window recheck under the lifecycle mutex, park/wake acknowledgement, stale-generation rejection, safe parked destruction, and Stage61's fail-closed partial-worker research rule. The synchronization repair appears once, not twice.
