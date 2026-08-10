# Open Images V7 optional control

Status: not-run-nonblocking.

The official V7 description/download pages, downloader source, validation
image metadata, validation bounding boxes, and class descriptions were
captured. The box annotations have SHA-256
d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2;
the validation image metadata has SHA-256
ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17.

The official downloader addresses the AWS bucket open-images-dataset, outside
this stage's network-domain allowlist. The allowlisted
storage.googleapis.com/open-images-dataset/ equivalent returned HTTP 403.
No Open Images image byte was downloaded, so O1 was not constructed and no
overlap or PTQ claim is made.

The official warning remains controlling: annotations are CC BY 4.0; images
are listed as CC BY 2.0; per-image license verification remains the user's
responsibility.
