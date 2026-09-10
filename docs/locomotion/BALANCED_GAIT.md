# Healthy gait balance using established rewards

Started September 10, 2026 at **16:45:40 UTC**.

User request: “ok, lets go with know rewards first, lets train”. The preceding review found that longer strides retained uneven stance timing. Keep existing healthy policies and videos for comparison.

Implement Isaac Lab Spot's completed air/contact-duration variance and body-motion penalties in the existing MuJoCo/mjbatch learner. Start with balance weight 20 and body-motion weight 1, retaining support weight 2 and stride weight 1. The variance formula clips each last completed interval at 0.5 seconds and uses sample variance across four feet. The body cost is `0.8 * vertical_velocity² + 0.2 * sum(abs(roll/pitch_angular_velocity))`. No fixed phase, mirrored actions or diagonal-pair requirement is introduced in this first run. [Pinned source and license](../../experiments/adaptive_locomotion/third_party/isaaclab_rewards/PROVENANCE.md).

The timing regularizer is only active for intact geometry and full-strength motors during motion commands. This uses privileged training state, adds no actor observations and does not claim deployed damage detection. Body stability remains useful across body conditions. Both new weights default to zero for reproducibility of previous runs.

First budget: 90 seconds from the 297.789-second longer-stride policy. Its existing 64/64 support-valid progress and persistent uneven duty factors justify this short extension under the user's standing ten-minute allowance; all ancestry remains counted. Use development seed 9137. Reserve final seed 20260914 for 64 trials and sixteen half-timestep trials. Acceptance: halve the mean left/right duty-factor gap and bring it below 10 percentage points, retain mean stride at least 30 cm, preserve comparable speed (within 10%), reduce body-height variation, retain foot-only support and inspect the video. Keep failed candidates if more refinement is needed.

## First attempt

The 20/1 balance/body-motion weighting regressed in 89.732 seconds: **0/32** development completions, little forward progress, and forbidden support. The checkpoint and complete development failure report are retained; it is not selected. Restart from the original longer-stride policy using balance weight 5 and body-motion weight 0.5 for a 120-second trial. This tests less aggressive regularization rather than extending the failed policy.
