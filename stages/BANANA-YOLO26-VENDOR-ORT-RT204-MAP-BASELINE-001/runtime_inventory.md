# runtime_inventory

Runtime under test: public SpacemiT ORT rt204 (`spacemit-ort.riscv64.2.0.4`).

| item | path | sha256 |
|---|---|---|
| rt204 archive | /data/banana-yolo26-spacemit-demo/.deps/runtimes/rt204/downloads/spacemit-ort.riscv64.2.0.4.tar.gz | bcf02bd12b8a1df969d6986658a9270c1121e5d58f5947d91ea5eba1bd6cd435 |
| board libonnxruntime.so.1 | /home/svt/banana-yolo26-trackb-rt204-map/runtime/rt204/lib/libonnxruntime.so.1 | e887a538b6cce9597b1905034b48f89763dd625b04bcd708ceb4b494df6df1ac |
| board libspacemit_ep.so.2 | /home/svt/banana-yolo26-trackb-rt204-map/runtime/rt204/lib/libspacemit_ep.so.2 | a59e29d2ed4c08ab57ad3e72c75a0b9a72020cb0e8f278ef2ef483725d04b47a |
| board yolo26_coco_predict | /home/svt/banana-yolo26-trackb-rt204-map/bin/yolo26_coco_predict | b1e9bb4ecec5886a705c2b56dc180cbbddd264ccade1aee374993d5f24a819db |

Board anchor excerpt:

```text
## board probe
bf3
Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64 riscv64 riscv64 GNU/Linux
90be7592-f6d9-4d69-ae40-a6c9d25a51ab
Architecture:        riscv64
Byte Order:          Little Endian
CPU(s):              8
On-line CPU(s) list: 0-7
Model name:          Spacemit(R) X60
Thread(s) per core:  1
Core(s) per socket:  8
Socket(s):           1
CPU(s) scaling MHz:  100%
CPU max MHz:         1600.0000
CPU min MHz:         614.4000
L1d cache:           256 KiB (8 instances)
L1i cache:           256 KiB (8 instances)
L2 cache:            1 MiB (2 instances)
scaling_governor
performance
scaling_cur_freq
1600000

```
