# Model training provenance

Classification: training-corpus-proven-coco-train2017.

The canonical FP32 ONNX model has SHA-256
d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2.
Its embedded Ultralytics metadata says that the model was trained on
ultralytics/cfg/datasets/coco.yaml, identifies 80 COCO classes, and identifies
the task as detection.

The matching packaged coco.yaml has SHA-256
7e05266d7d0b0247ac603602c155507fe54a86b91e470f70a7f79502dfa8ab61.
It maps training to train2017.txt with 118,287 images and validation to
val2017.txt with 5,000 images. The ONNX metadata, class map, dataset file, and
accepted model lineage jointly establish the training-corpus classification.
The embedded build-machine path is provenance text only and is not exported
as an operational dependency.
