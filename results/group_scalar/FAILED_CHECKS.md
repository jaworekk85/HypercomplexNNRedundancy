# Failed check retained

The first diagnostic invocation stopped at seed 2026, block 1 with
`AssertionError: ['calibration_snapshot_shape']`. The driver incorrectly
expected 64 spatial positions at block 1. A 32x32 image after one 2x2 pooling
has 16x16 = 256 positions. Blocks 1/2/3 therefore have 256/64/16 positions.

Corrected only the shape assertion to `(32 // (2**block))**2`. No stored
activation, score definition, label, model, threshold or primary block was
changed. No successful-result summary had been written. The failed driver
source hash/configuration remain in `failed_run_protocol.json`; the corrected
invocation writes its own protocol and repeats all checks.
