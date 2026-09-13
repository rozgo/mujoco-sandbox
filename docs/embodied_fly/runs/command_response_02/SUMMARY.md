# Previous wing-command response

Frozen sensory probe; no physics or training. Full joint measurements and
other inputs are held fixed while previous wing commands change by ±0.02.
All action replays match their captured histories within 7e-7. The largest
ground response spectral radius is 0.2581. This does not show strong
self-amplification in the sampled local command map. It is not a complete
closed-loop stability proof. Do not introduce a new invariance penalty based
on this probe alone. Each checkpoint uses its own history; frame-zero physical
inputs and zero neural state are matched.
