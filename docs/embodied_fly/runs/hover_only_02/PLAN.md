# Second bounded hover-only PPO interval

Decision after the fixed first pilot: all three predeclared ten-second starts
now remain airborne, versus two before training. Position holding is still poor:
nominal peak drift 58.462 mm versus 61.346 mm before; settled height span worsens
from 5.802 to 6.055 mm. This is a recovery-basin gain, not reference-quality hover.
No promotion or harder reset curriculum is justified.

The user's standing authorization permits another short interval when the
measurements justify it. Continue for one more requested 600 seconds from
`hover_only_01/actor.pt`, SHA256
`6d5681aa96c2b6bc154d63b21a53d0236180bb607f6b1f317b11a2543d3f21a9`.
Keep the first pilot's 64 worlds, 16 physics threads, plant, observations, reward,
500 Hz actions, 512/128 rollout/sequence, two epochs, actor/critic rates, noise,
discounts, independent critic and KL limit. Seed 120102. Resume saved actor,
critic, Adam states and exploration. Set critic warmup to zero because the
first interval already completed it. No PID or imitation enters the learner.

Repeat the same nominal/lower/lateral deterministic ten-second evaluation and
PID world. Preserve the first video and create a second comparison. Do not
advance to forward flight unless hover quality supports it. Compare complete
errors and failures, not training return alone. If a second interval does not
materially improve holding, report the limitation instead of spending hours on
an unchanged recipe.
