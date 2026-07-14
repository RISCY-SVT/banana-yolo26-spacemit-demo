# Dense dispatch policy

Prepare chooses exact direct-1x1, packed 3x3 P3 stride-2, RGB stem, small-N, or the accepted M12xN16 fallback from static tensor descriptors. M12/N16, A-stationary, and spatial partition remain the general winner; M8, weight-stationary, output-channel, and 2D partitions were exact but slower.
