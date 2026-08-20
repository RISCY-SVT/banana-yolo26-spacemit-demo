# Recovery Event Contract

A current-stage host reboot may be claimed only when at least one of these is
available and tied to the stage interval:

- host boot identity before and after;
- OS or hypervisor boot-boundary evidence;
- explicit operator attestation of a current-stage event.

An ordinary maintenance reboot between stages is an environment lifecycle
event, not a stage recovery event. A partial output root, failed command, or
missing completion marker is not by itself evidence of an operating-system
reboot. Tool defects and isolated partial roots must be classified directly.
