# RT204 and RT205 performance

The stable, matching primary-QDQ CPU surface used CPU0-3, ORT_ENABLE_ALL,
sequential execution, intra=4/inter=1, and warmup/runs/repeats 10/100/5.

- RT204 CPU: `1023677.578412 us` (stddev `317.786422 us`).
- RT205 CPU: `1024818.557872 us` (stddev `620.595993 us`).
- RT205 is `0.111459%` slower than RT204 on this surface.
- The accepted Stage45 B120 CPU baseline is `461603.297250 us` and remains
  materially faster than either release package CPU runtime.

The bounded intra-thread scouts were RT204 `2490869.192150 / 1508708.831175 /
1023677.578412 us` and RT205 `2485203.749075 / 1510556.013500 /
1024818.557872 us` for intra 1/2/4 respectively. Intra=4 is the stable selected
CPU resource setting; semantic oracle sessions remain single-threaded.

There is no accepted RT204 or RT205 SpacemiT-EP full-model latency for the
primary model: RT204 fails the historical first-Conv compiler gate, and RT205
fails the same gate before aborting. QOperator rows are diagnostic because RT204
does not preserve the accepted output and RT205 SIGILLs. No hidden fallback row
is promoted as accelerated INT8.

Separate RT205 EP diagnostic controls run at `445504.354768 us` for FP32
and `368527.093944 us` for the body/head FP16 model. Their synthetic e2e
`output0` arrays are not cross-runtime exact, so these are timing diagnostics,
not accepted semantic or production paths. Both remain far above 50 ms.
