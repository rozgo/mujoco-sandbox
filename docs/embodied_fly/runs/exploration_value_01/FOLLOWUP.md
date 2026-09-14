# Input scaling follow-up and limits

The source-pinned offline replay is `python -m embodied_fly.critic_scale
outputs/embodied_fly/exploration_value_01 --output NEW_DIRECTORY`.
It repeats four matched fits: original/scaled inputs crossed with original/scaled
reward targets. Each fresh MLP uses the same seed, batches and 1,000 updates.
Input statistics use training streams only. Outputs remain physical reward units.
Full machine-readable measurements are in input_scale_followup.json.

With original targets, standardizing inputs improves held-out explained variance
from 0.3469 to 0.4718 and RMSE from 1.1555 to 1.0633 on the complete mixed set.
On the narrower .001/.003 noise subset it worsens explained variance from 0.7158
to 0.1889 and RMSE from 0.4809 to 0.5788. Thus the overall result partly reflects
states leading to failure at .006. There is no demonstrated universal benefit.
The chosen bounded PPO trial tests a training hypothesis, not an established fix.
It adds more critic updates and modest exploration as explicitly declared.

Neither these diagnostic critics nor their four-second return targets are loaded
into PPO. PPO starts with a fresh critic, learns from its own timeout-aware GAE
returns, and calibrates its inputs only on its first collected training rollout.
The motor actor and physics are unchanged before PPO. Eight focused tests pass;
the full revised package passes 184 tests in 157.62 seconds (45 dependency warnings).
