# Stage65D workspace preflight

- Status: `pass`.
- DEV-001C packet: `ce214eb6e906586ffc98d5da823d4406bf1ea627d5e8ae65a823e259efdb38f1`, 44 files, 269690 bytes.
- Frozen B2/C2/common tail and ORT archive/core/EP: exact.
- Banana, XSlim, protected main, custom executor, and accepted ncnn state: exact.
- Board: `bf3`, serial `92262f3b0dc4`, boot `0a0691d1-7502-44c3-903b-444dba83c1d9`, `Bianbu 2.2.1`, kernel `6.6.63`.
- Board `/data`: NVMe, writable, 436358606848 bytes free. Root filesystem is eMMC and project writes there are forbidden.
- `/usr/bin/time` was installed during fail-closed preflight before Stage roots were created; all mandatory commands are now present.
- No current-stage reboot is claimed.
