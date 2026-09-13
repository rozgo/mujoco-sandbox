# Smaller frozen exploration probe

Audit01: all eight mean-action worlds remain airborne; three of eight sampled
worlds fail. The five sampled survivors average about the same lift as the mean
policy but usually bob more. Thus the current noise causes episodic destabilization,
not simply useful extra average thrust. Full-graph pre-update replay mean KL is
1.09e-5, well below .03; maximum individual log-probability difference is .193.
Do not claim bitwise likelihood equality or a long-gradient audit.

Repeat the same checkpoint, nominal starts,16 worlds, seed120201 and5 seconds,
changing only sampled exploration standard deviation/floor from .003 to .001.
This diagnostic overrides the saved distribution explicitly and never changes
the checkpoint. Retain all eight stochastic outcomes and all eight deterministic
controls. A successful probe informs a declared training-distribution change;
it does not prove robustness or learned-hover improvement.
