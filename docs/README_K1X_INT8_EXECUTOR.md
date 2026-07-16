# YOLO26 K1X INT8 Executor

Version 0.9.0 is the frozen engineering-handoff release for one exact target:
YOLO26n-640 on Banana-Pi BPI-F3 / SpacemiT K1X under `K1X_INT8_V1`.

Start here:

- [HANDOFF_EN.md](HANDOFF_EN.md)
- [HANDOFF_RU.md](HANDOFF_RU.md)
- [QUICKSTART_RU.md](QUICKSTART_RU.md)
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- [RELEASE_PROFILES.md](RELEASE_PROFILES.md)
- [SYSTEM_PROFILE_O2.md](SYSTEM_PROFILE_O2.md)
- [PERFORMANCE_AND_ACCURACY.md](PERFORMANCE_AND_ACCURACY.md)
- [TROUBLESHOOTING_HANDOFF.md](TROUBLESHOOTING_HANDOFF.md)
- [CURRENT_GRAPH_FREEZE.md](CURRENT_GRAPH_FREEZE.md)

The release path contains no ONNX Runtime or Python call and requires no
stage-numbered environment variables. It installs a C ABI1 library, CLI,
healthcheck, CMake/pkg-config metadata, and C example. Historical tools and
candidate kernels remain available only through the explicit research build.

This is an optimized exact reference and engineering handoff. It is not a
production certification, camera service, or 20 FPS claim.
