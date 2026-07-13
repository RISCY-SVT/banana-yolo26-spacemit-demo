# Cluster1 non-IME report

A second pool pinned to CPU4-7 executed only common RVV/integer LUT, Add, and Concat work. All
outputs were exact and the CPU4-7 IME count was zero. The complete slice mean changed from
17833.736802 us on cluster0 to 18266.834038 us on cluster1
(2.428528%). Standalone rows likewise did
not provide a stable end-to-end win, so cluster1 offload is proven but not selected.
