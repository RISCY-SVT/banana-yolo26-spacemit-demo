# Indexed-load SIGILL forensics

The first malformed I1 attempt trapped on CPU0 and CPU4 at PC `0x12488`, raw instruction `0x4a032157`, with `vill=1`. After correction, I0-I6 execute exactly on CPU0 and CPU4. The selected route has no SIGILL; the malformed attempt remains narrow diagnostic evidence.
