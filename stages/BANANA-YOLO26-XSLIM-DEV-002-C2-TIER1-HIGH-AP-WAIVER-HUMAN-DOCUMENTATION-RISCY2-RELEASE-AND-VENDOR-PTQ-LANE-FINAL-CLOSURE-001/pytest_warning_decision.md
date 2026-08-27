# Pytest Warning Decision

The pinned pre-change suite passed 207 tests and emitted eight `onnxconverter_common.float16` `UserWarning` messages. The values were individually observed as:

1. `-2.5924412838662647e-08 -> -1e-07`
2. `-1.7196978063793722e-08 -> -1e-07`
3. `+2.351264782873841e-08 -> +1e-07`
4. `+8.609106316725956e-08 -> +1e-07`
5. `+9.168664405478921e-08 -> +1e-07`
6. `-3.4068833087985695e-08 -> -1e-07`
7. `+1.0627975832733227e-08 -> +1e-07`
8. `-1.624368728414538e-08 -> -1e-07`

These are deterministic third-party FP16 minimum-magnitude conversions in the FP16 integration fixture, not downstream runtime warnings. The test now fixes the torch seed, captures only this exact warning shape and fails on any different category/message. No warning category is globally suppressed. Final result: 212 cases plus 65 subtests pass with zero uncaptured warnings.
