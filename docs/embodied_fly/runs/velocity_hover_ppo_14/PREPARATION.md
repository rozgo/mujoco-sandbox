# Reward and recovery checks before run 14

The coordinated reward audit passed in **50.042555 s**, with no learning.
Recorded actions reproduced every body position exactly on the original plant.
On every paired case, the candidate scores the retained run 11 midpoint above
both run 12 snapshots and both run 13 snapshots. Synthetic recovery also beats
early failure under both summed and discounted return.

| Saved flight | Previous reward | Coordinated reward |
| --- | ---: | ---: |
| Retained run 11 midpoint | 33.8093 | **24.3885** |
| Run 13 midpoint | 40.6615 | 23.0740 |
| Run 13 final | 41.4673 | 22.4039 |

Scores are ten-second sums. Their numerical magnitudes across different
objectives are not comparable; their ordering is the relevant check. The
coordinated reward prefers the better combined hover. This supports trying PPO,
but does not establish that the gradient or representation will solve control.

The new recovery bank comes from frozen run 13 final, with 24 training histories
and six withheld ones at 2/4/6 seconds. It explicitly uses the proposed scorer.
The stored reward history contains raw velocity samples, never privileged
teacher commands. Physics, sensing, full neural memory and graph are unchanged.

- Bank SHA256: `50ae2f19006947a4f1dd6b4acbafe70c08a18a8561c7745e411c758149ddd636`.
- Setup **9.850156 s**, collection **23.272318 s**, restoration audit **1.956507 s**.
- 30,640 collection transitions; 1,920 neural and 1,920 recorded-action audit
  transitions. These costs are separate from PPO training.
- All initial states exact; recorded-action observation, joint position and
  reward errors zero. Closed-loop body-position difference **0.633 micrometres**
  over 128 ms, within the existing one-micrometre bound. No tolerance changed.
- Full suite **266 passed**, 45 upstream warnings, **171.60 s**. Focused tests
  **23 passed** in **4.01 s**. Lint passed.

The learning run uses the declared reward transition and a critic-only first
rollout. A new recovery bank was necessary because the experimental parent is
now run 13 final. Run 11 remains the preferred overall policy until physical
evaluation satisfies the combined gate.
