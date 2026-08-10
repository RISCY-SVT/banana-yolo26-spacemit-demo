# Hash contract

Decoder: Pillow `12.3.0`. Orientation is normalized with `ImageOps.exif_transpose`, then converted to contiguous row-major RGB8. The canonical digest hashes `stage65b-r1-rgb8-v1\0`, little-endian uint64 width and height, and the RGB bytes. Exact JPEG SHA-256 is kept separately. A 64-bit 9x8 grayscale dHash with Hamming distance <= 1 is a review warning only.
