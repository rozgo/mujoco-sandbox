# First wing-motion decoder warm start

Source `39960b2`, seed 62001, readout01 parent. Eight valid two-second reference episodes produced 8,000 physical transitions / 16 aggregate simulated seconds in 59.446307 seconds. Full-graph feature replay took 9.654989 seconds total, including 5.738637 seconds of neural replay across eight sequences.

RTX 4090 decoder fitting took 60.000525 seconds after 3.929016 seconds of setup: 48,922 updates / 50,096,128 reused samples from 14,000 distinct training frames. Only the existing 230,572 decoder parameters update; upstream actor tensors remain fixed. No live physics worlds run during fitting. Peak CUDA allocation: 116,486,144 bytes.

Excluded flight wing MSE improved 0.366067 → 0.000974, and startup MSE 0.446364 → 0.005563. Both independent two-second student flights still failed. Checkpoint/model/state hashes and causal previous-action feedback were verified after transfer. No promotion. The next phase-zero corpus and complete flight-posture curriculum are declared in the learning journal.
