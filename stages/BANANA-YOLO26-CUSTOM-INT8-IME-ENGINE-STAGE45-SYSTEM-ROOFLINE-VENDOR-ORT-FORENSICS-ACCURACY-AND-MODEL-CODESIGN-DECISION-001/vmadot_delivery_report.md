# vmadot delivery report

Stage35 register controls reproduced: dependent `3.75144 ns/vmadot`, four-way
independent `0.93811`, six-way independent `0.625309`; all exact and trap-free.
The new packed standalone matrix shows delivery loss and register-block benefit:

- M4xN16 model5 geometry: `24.746637 GMAC/s`, `9533.804644 us`.
- M8xN16: `43.395240 GMAC/s`, `5436.762213 us`.
- M12xN16: `54.360135 GMAC/s`, `4329.271337 us`.

Every shape passed a scalar grouped oracle. These are standalone packed-compute
ceilings, not integrated kernels or authorization to change dispatch. M12xN16 is
the best diagnostic shape, but spilling and full operator dataflow remain unknown.
