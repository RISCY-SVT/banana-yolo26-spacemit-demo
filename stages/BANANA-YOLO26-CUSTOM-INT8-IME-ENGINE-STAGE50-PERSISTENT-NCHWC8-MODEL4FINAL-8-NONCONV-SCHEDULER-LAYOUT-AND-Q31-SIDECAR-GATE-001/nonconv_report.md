# Non-Conv optimization

The explicit indexed RVV LUT plus parallel Add/Concat and active-worker completion reduced the short slice scout from 34.714 ms to 27.038 ms while preserving all bytes. Compatible Concats are producer-direct/view operations; the remaining quant-domain rescale remains a measured pass and is not mislabeled zero-copy. An exact explicit vector Add route was not selected.
