
# Non-MAC operator LUT

Resident-int8 LUT, Add+SiLU, and Concat rows come from the integrated slice's
one diagnostic profile. MaxPool, Resize, Softmax, TopK, and GatherElements remain
conservative B120 profile rows. They are mapping evidence, not optimized custom
implementations; profile perturbation is kept out of headline wall timing.
