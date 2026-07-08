# STAGE_REPORT

classification: track-b-pass-yolo26-value-confirmed
stage_id: BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: b54c8767e691dc57cbd035a13d2d2d348d2f5366
end_head: pending-local-commit-see-final-response
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
track_b_executed: yes
full_coco_status: pass
variants_measured: fp32_e2e_rt204, fp16_keepio_rt204
log_dir: /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
result_packet: /exchange/results/outbox/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001

## Headline mAP table

```text
variant	description	AP	AP50	AP75	AP_small	AP_medium	AP_large	AR1	AR10	AR100	images	full_coco_mean_total_ms	full_coco_std_total_ms	full_coco_fps	objects_total
fp32_e2e_rt204	YOLO26n FP32 e2e 640 rt204 SpaceMIT EP	0.40472980774853884	0.5712208600698203	0.43502779558879534	0.19781524372306147	0.441532191383899	0.5869564056200277	0.3294420897289608	0.542685219567277	0.589989700687529	5000	526.1296387128	7.924237595831371	1.900672242009679	550555
fp16_keepio_rt204	YOLO26n FP16 body/head keep-I/O 640 rt204 SpaceMIT EP	0.40474751048699903	0.5714174623514189	0.4352412466360423	0.1981407609388083	0.4414811887105845	0.5869213719789296	0.3292910025232672	0.5427411723817106	0.5899412840000278	5000	397.1280944456	4.3495424637431475	2.5180792142042305	550151

```

## Headline FPS / latency table

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

## Proven

- Full COCO val2017 bbox mAP was measured from board-generated rt204 predictions.
- YOLO26n FP32 and FP16 keep-I/O have matching AP class (`AP≈0.405`).
- FP16 keep-I/O is the better public rt204 speed path among measured YOLO26 variants.
- Loader proof resolves rt204 runtime from the stage board workspace.
- No /data/ncnn mutation, no YOLO11 repo mutation, no INT8/custom-engine claim.

## Broken

- Public vendor rt204 YOLO26 speed remains below imported YOLO11 production INT8 reference.
- INT8 vendor acceleration remains blocked by previously documented Q/DQ Conv issue; it was not rerun as an accepted path.

## Unknown

- Production camera/full-image behavior for YOLO26 remains unknown and unclaimed.
- Custom engine full-model mAP/FPS remains unknown.

## Non-claims

This is not a production release, not full custom engine inference, not camera readiness, not COCO/mAP for the custom INT8 engine, and not a default-backend claim.
