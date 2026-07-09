# Fallback Policy

Fallback sections are ONNX Runtime CPU cuts, not custom acceleration.

| section | implementation | status |
|---|---|---|
| `images -> /model.4 input` | ORT CPU prefix cut | exact vs full ORT boundary |
| `/model.4 input -> /model.4 output` | ORT CPU fallback cut | exact vs full ORT boundary |
| `/model.4 input -> /model.4 output` | custom C++ runner on board | exact vs ORT boundary |
| `/model.4 output -> output0` | ORT CPU suffix cut | exact vs full ORT final output |

Fallback policy rules:

- Do not treat ORT CPU fallback time as custom-engine speed.
- Do not use SpacemiT EP as oracle.
- Do not change default app/backend behavior.
- Do not make full-model FPS or production claims.
