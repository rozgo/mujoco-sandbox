# Smaller frozen exploration preserves all sampled starts

The paired five-second probe changes only sampled latent standard deviation and
floor to .001. Same pilot04 actor, 16 worlds, seed 120201, nominal starts and accepted
plant. All eight sampled paths remain airborne. Their height spans after 1 s are
5.975–6.458 mm, versus 5.566 mm for the deterministic controls. Mean survivor lift
is 1.0110 body weights. This is a local exploration diagnostic, not a trained-policy
improvement or proof of broad robustness.

Unchanged-parameter forward replay has mean approximate KL 2.6099e-6 and maximum
individual log-probability difference .03064. The tiny change in deterministic
trajectories across runs is retained; matching seeds does not imply bitwise GPU
replay. Source 89f10e0; setup 7.050627 s, capture 21.843646 s, replay .826064 s;
40,000 physical transitions, zero optimizer updates.

An offline [score diagnostic](score_diagnostic.json) on the earlier complete
PID/PPO capture confirms that the declared reward favors the reference. After
1 s, the PID's height/position scores average 1.999/.993; PPO's average .806/.051.
Velocity scores also favor PID. These four terms omit survival, orientation and
effort and are not total episode returns. They identify the large positional gap
without changing rewards or performing another physical rollout.

The result motivates an explicit exploration reset for the next PPO run, paired
with a smaller actor step for the narrower distribution. It does not justify
changing the plant or claiming smooth hover. See [raw results](evaluation.json),
[summary statistics](SUMMARY.json) and [predeclared plan](PLAN.md).
