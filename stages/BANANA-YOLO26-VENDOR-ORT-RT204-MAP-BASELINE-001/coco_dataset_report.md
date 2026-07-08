# coco_dataset_report

Full COCO val2017 was available locally and on the board; no dataset download was performed.

| item | path | status |
|---|---|---|
| host val2017 | /data/datasets/coco2017/val2017 | 5000 images |
| host annotations | /data/datasets/coco2017/annotations/instances_val2017.json | present, 20M |
| board val2017 | /home/svt/datasets/coco2017/val2017 | 5000 images |
| board annotations | /home/svt/datasets/coco2017/annotations/instances_val2017.json | present |

Evaluation used `pycocotools==2.0.11` in `.deps/venvs/trackb_coco_eval` on host.
