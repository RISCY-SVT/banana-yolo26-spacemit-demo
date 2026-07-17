# UI Sidecar Decision

A separate HighGUI owner thread was optional and was not selected. The matched
performance preset already sustains one display call per processed frame, and
Stage59 did not produce the required thread-ownership, frame/result pairing,
reconnect, signal, and long-soak proof for a second UI queue.

All HighGUI calls therefore remain on the application consumer thread. This is
the simpler exact pairing contract; no unmeasured UI throughput gain is claimed.
