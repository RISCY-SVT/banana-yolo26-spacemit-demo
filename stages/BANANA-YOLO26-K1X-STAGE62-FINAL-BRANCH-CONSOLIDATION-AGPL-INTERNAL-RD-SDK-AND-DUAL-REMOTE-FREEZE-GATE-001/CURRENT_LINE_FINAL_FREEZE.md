# Current Line Final Freeze

The YOLO26 K1X current executor line is frozen for maintenance and evidence. R640 remains the accepted exact default. The other eight Q0 profiles are explicit research opt-ins and are not deployment-promoted. Future PTQ, training, model change, or co-design requires a new branch/project and separate authorization.

- Stable R640: `v0.9.3-r640` / `d0e3611c8d99dfade049bd261cb557509222a456`
- Stage61 research: `stage61-q0-final` / `fa668ccaf7938336bd10313455ab81557b33e020`
- Integrated release source: `9f88644aef6a9eb304cae3e95b62da6a0aa22cc3`
- R640 merged pure mean: 131.751 ms
- R384 diagnostic: 21.017 pure FPS, 6.420 AP loss, not promoted
- R768: 5.070 pure FPS, +0.281 AP point estimate, mixed effect, not promoted
