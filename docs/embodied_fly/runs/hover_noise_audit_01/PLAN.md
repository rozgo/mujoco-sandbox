# Frozen hover diagnostic before continued PPO

The prior goal turn made concrete progress: four physical pilots, two identified
training issues, matched PID/PPO videos and synchronized archival evidence. The
full fly-survival objective remains open; usable motor control precedes utility.

Keep the accepted 1 kHz instantaneous-wing-force plant and pilot04 actor frozen.
Run eight independent mean-action worlds and eight sampled-action worlds, using
the recorded exploration distribution, for five seconds from the same nominal
start. No PID, reset, optimizer, clipping change or extra controller is added.
Retain physical failures, applied noise, lift, position and bounded reward.

Replay the first128 recorded observations/actions through the unchanged full
graph with gradients enabled, checking likelihood agreement. This is a forward
numeric audit, not a test of long-horizon gradient accuracy. It tests whether
collection and replay disagree before any parameter update. No training video
is required for this frozen measurement; subsequent trained videos keep the
PID comparison and automatically open after verification.

Use the measured noise effect to choose the next bounded PPO continuation.
Do not silently alter the physical body, wing force law, action rate, observed
inputs, policy architecture or fixed graph. If the noise effect is modest,
continue pilot04's actor/critic/Adam at the existing3e-7 actor learning rate.
