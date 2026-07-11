# Model5 Selection Decision

Model5 was selected for the implementation gate because it is the only authorized block directly adjacent to the current model4 custom island. Model6 has the largest isolated board ORT time but cannot improve the contiguous island without first crossing model5.

The implemented candidates reuse the existing signed-storage Stage37 four/six-accumulator MMT4D `smt.vmadot` route. Four accumulators were retained as the bounded candidate; six accumulators did not reverse the result.

Decision: exact implementation retained only as uncommitted experimental evidence; performance selection rejected.

Reason: same-session exact custom model5 Conv plus postactivation is `26579.802 us`, versus `17862.861 us` for the equivalent isolated board ORT model5 segment. The custom path is `48.799%` slower. The full hybrid scaffold gate was therefore short-circuited.
