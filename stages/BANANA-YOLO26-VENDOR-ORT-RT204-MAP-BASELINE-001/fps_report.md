# fps_report

This report separates metric classes. These are selected runtime/model measurements, not model FPS production claims.

## Stable app/perf matrix

```text
variant	class	provider	model	threads	pin	warmup	runs	repeats	mean_ms	std_ms	fps	notes
fp32_e2e_rt204	app_forward	spacemit	models/yolo26n_640_e2e_fp32.onnx	4	cluster0	10	100	5	575.875494	4.298396	1.736486	single fixed COCO image 000000000139.jpg
fp32_e2e_rt204	app_full	spacemit	models/yolo26n_640_e2e_fp32.onnx	4	cluster0	10	100	5	517.638308	0.712594	1.931851	single fixed COCO image 000000000139.jpg
fp16_keepio_rt204	app_forward	spacemit	models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx	4	cluster0	10	100	5	380.36272	0.895146	2.629069	single fixed COCO image 000000000139.jpg
fp16_keepio_rt204	app_full	spacemit	models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx	4	cluster0	10	100	5	395.546444	0.872901	2.528148	single fixed COCO image 000000000139.jpg
fp32_e2e_rt204	perf_test_forward	spacemit	fp32_e2e_rt204	4	taskset-0-3	n/a	100	1-of-5	555.431444	per-repeat-row	1.8004022112943248	one perf_test repeat; five rows recorded
fp32_e2e_rt204	perf_test_forward	spacemit	fp32_e2e_rt204	4	taskset-0-3	n/a	100	1-of-5	577.331854	per-repeat-row	1.7321060548999259	one perf_test repeat; five rows recorded
fp32_e2e_rt204	perf_test_forward	spacemit	fp32_e2e_rt204	4	taskset-0-3	n/a	100	1-of-5	560.605162	per-repeat-row	1.7837866430491414	one perf_test repeat; five rows recorded
fp32_e2e_rt204	perf_test_forward	spacemit	fp32_e2e_rt204	4	taskset-0-3	n/a	100	1-of-5	584.817698	per-repeat-row	1.709934571781718	one perf_test repeat; five rows recorded
fp32_e2e_rt204	perf_test_forward	spacemit	fp32_e2e_rt204	4	taskset-0-3	n/a	100	1-of-5	568.303655	per-repeat-row	1.7596226791819594	one perf_test repeat; five rows recorded
fp16_keepio_rt204	perf_test_forward	spacemit	fp16_keepio_rt204	4	taskset-0-3	n/a	100	1-of-5	378.172982	per-repeat-row	2.6442925528720083	one perf_test repeat; five rows recorded
fp16_keepio_rt204	perf_test_forward	spacemit	fp16_keepio_rt204	4	taskset-0-3	n/a	100	1-of-5	380.508959	per-repeat-row	2.6280590150309706	one perf_test repeat; five rows recorded
fp16_keepio_rt204	perf_test_forward	spacemit	fp16_keepio_rt204	4	taskset-0-3	n/a	100	1-of-5	380.924158	per-repeat-row	2.6251944881899565	one perf_test repeat; five rows recorded
fp16_keepio_rt204	perf_test_forward	spacemit	fp16_keepio_rt204	4	taskset-0-3	n/a	100	1-of-5	377.797331	per-repeat-row	2.6469218227483986	one perf_test repeat; five rows recorded
fp16_keepio_rt204	perf_test_forward	spacemit	fp16_keepio_rt204	4	taskset-0-3	n/a	100	1-of-5	378.630908	per-repeat-row	2.6410944771576865	one perf_test repeat; five rows recorded

```

## Perf-test aggregate

| variant | perf_test_mean_ms | perf_test_std_ms | fps |
|---|---:|---:|---:|
| fp32_e2e_rt204 | 569.297963 | 10.716710 | 1.756549 |
| fp16_keepio_rt204 | 379.206868 | 1.267436 | 2.637083 |


## Full COCO generation timing

| variant | mean total ms/image | stddev ms | derived fps | images |
|---|---:|---:|---:|---:|
| FP32 e2e rt204 | 526.129639 | 7.924238 | 1.900672 | 5000 |
| FP16 keep-I/O rt204 | 397.128094 | 4.349542 | 2.518079 | 5000 |
