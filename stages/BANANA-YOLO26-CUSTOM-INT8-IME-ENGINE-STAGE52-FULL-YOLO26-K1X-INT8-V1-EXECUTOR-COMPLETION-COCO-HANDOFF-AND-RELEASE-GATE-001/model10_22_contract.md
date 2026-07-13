# Model 10 through model 22 contract

The package represents the complete accepted graph from model9 through
model22. Dense Conv uses the exact Q62 E2c route; eight depthwise/grouped rows
use an internal direct exact path. Concat, Split, SPPF pooling, Resize, and
activation transformations use package-defined integer tables or exact view
and placement rules. No operation has a float or ORT fallback.
