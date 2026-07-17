# Supported Platform

The 0.9.2 runtime is validated only on the following platform contract:

- Banana-Pi BPI-F3 with SpacemiT K1X/X60 CPUs;
- Bianbu 2.2.1 user space;
- Linux 6.6.63 board kernel;
- RV64GCV plus the approved named `smt.vmadot` symbols;
- CPU0-3 IME workers and CPU4 controller;
- NVMe `/data` for deployment, logs, media, and model assets;
- the system sonames listed in `required-system-sonames.tsv`;
- bundled OpenCV 4.13 camera-demo libraries where present.

The archive is not root-filesystem independent. "Offline" means that, after
copying the archive to the tested board image, it does not need network access
to execute the included package and demo. It does not mean that the archive
contains glibc, the dynamic loader, V4L2/kernel drivers, GUI services, or every
operating-system dependency.

Run `scripts/verify-system-dependencies.sh` from an extracted release before
using it on another board image. A successful check is a compatibility preflight,
not production certification.
