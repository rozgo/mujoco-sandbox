# Paired actor-step diagnostic

The bounded-reward pilot fails the three starts at 1.426, 1.518 and 1.842 s.
Its median maximum observed approximate KL is 1.334 against a .03 threshold.
The current PPO guard stops later updates after detecting divergence; it cannot
undo an already excessive step. Pilots 01/02 also overshoot after one accepted
update, with median observed KL 1.287/1.057. This is a measured optimization
limitation, not evidence that the accepted physical plant cannot hover.

Repeat pilot 03 from the identical pilot-01 actor/Adam checkpoint, seed 120103,
fresh critic, four warmup rollouts, bounded reward, 64 worlds and every other
setting unchanged. **Only actor learning rate changes: 3e-6 -> 3e-7.** Requested
budget remains 300 seconds. Record the actual number of actor updates and KL;
compare the same three physical ten-second cases and matching PID world.

Keep all earlier weights and videos. Do not promote a checkpoint that fails
physical review merely because its optimization statistics improve. This is the
last bounded diagnostic in this review round; deliver the comparison and evidence
before another training-method change. The first four critic-only warmup rollouts
should provide a matched stochastic initialization check, separate from timing.
