# Worker stability report

S0 shared the CPU0-3 set between controller and workers and showed periodic wake/scheduling excursions: 5603.697698 us mean and 8768.343200 us p95. S1 pinned the controller to CPU4 while all IME workers remained on CPU0-3: 5145.442776 us mean and 7663.699800 us p95. S1 was selected. CPU4 executes controller work only; no IME instruction runs there. S2 spin/epoch was not needed after S1 met the p95 ceiling.
