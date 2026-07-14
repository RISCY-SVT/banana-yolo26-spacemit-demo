# Structural fusion decision

Selected direct physical LUT2, attention split/transpose, C8 Resize/Concat, resident model9, and producer-direct compatible Concats. Conv-to-LUT1 fusion was exact but rejected because its alias-safe complete-model arm regressed wall time.
