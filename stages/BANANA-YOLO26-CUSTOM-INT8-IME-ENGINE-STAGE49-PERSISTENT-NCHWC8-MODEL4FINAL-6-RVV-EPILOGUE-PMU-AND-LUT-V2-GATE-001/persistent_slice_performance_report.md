# Persistent slice performance report

Under the stable 10/100/5 protocol, the exact custom internal slice measured 26710.414338 us mean and 26845.251100 us p95. The equivalent B120 ORT CPU diagnostic cut measured 42036.659040 us mean and 42078.047838 us p95. Custom mean is 36.459236% lower, satisfying the `<=0.90x` mean and p95 gate with zero internal conversions.

Scalar diagnostic adapters remain expensive: 30184.610616 us entry and 7855.612998 us exit; the derived custom-with-adapters surface is 64750.637952 us. This does not weaken the persistent internal result, but it prohibits an end-to-end claim.
