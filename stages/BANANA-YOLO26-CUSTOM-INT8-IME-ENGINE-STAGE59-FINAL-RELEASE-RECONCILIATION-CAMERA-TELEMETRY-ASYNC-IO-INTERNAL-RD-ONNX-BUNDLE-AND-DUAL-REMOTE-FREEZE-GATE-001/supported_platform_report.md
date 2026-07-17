# Supported Platform Report

The 0.9.2 archives are validated for Banana-Pi BPI-F3 / SpacemiT K1X running
Bianbu 2.2.1 and Linux 6.6.63. IME workers are CPU0-3 and the controller is
CPU4. Deployment, model, logs, media, and temporary artifacts remain on NVMe
`/data`.

The archives are offline-capable on the tested board image after transfer, but
not root-filesystem independent. They rely on the board dynamic loader, glibc,
libstdc++, pthread/math libraries, V4L2/kernel drivers, GUI services when used,
and every soname listed in `required-system-sonames.tsv`. The dependency verifier
is a compatibility preflight, not production certification.
