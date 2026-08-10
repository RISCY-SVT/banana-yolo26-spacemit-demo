# Dataset authority capture

Captured at 2026-08-08T19:30Z from the authorized official surfaces.

| Surface | URL | HTTP | Identity |
|---|---|---:|---|
| COCO overview | https://cocodataset.org/ | 200 | ETag 654c452e-2192, 8,594 bytes |
| COCO annotations | https://images.cocodataset.org/annotations/annotations_trainval2017.zip | 200 | 252,907,541 bytes |
| COCO train2017 | https://images.cocodataset.org/zips/train2017.zip | 200 | 19,336,861,798 bytes |
| Open Images V7 description | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | 200 | ETag dd8a2dd888823e94ffba1e1018b4a29f |
| Open Images V7 downloads | https://storage.googleapis.com/openimages/web/download_v7.html | 200 | ETag a84196d98a81075c06924f5b82beac64 |
| Official Open Images downloader | https://raw.githubusercontent.com/openimages/dataset/master/downloader.py | 200 | ETag ef271fff74dc7576255a97e3638df074a69993ca4edffa9937ba6125e0892eb6 |

The COCO object endpoint presented an Amazon S3 certificate that did not
cover images.cocodataset.org. Download remained on the explicitly authorized
official hostname and used the captured leaf public-key pin
sha256///Q5PRP4LtYp2a+3iFjt2mIJ3GPE6PhOJjrWnu8eltKc=. Both archives then
passed complete ZIP integrity and path-safety checks. This records the
transport exception; it does not claim a normally valid hostname chain.

Raw HTML, headers, certificate chain, and downloader source are retained under
the stage root in metadata/.
