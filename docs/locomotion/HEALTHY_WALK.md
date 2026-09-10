# Dedicated healthy-dog walking baseline

Follow-up started September 10, 2026 at 14:48:45 UTC.

User request: "do we have a normal dog trained?" / "if not lets train a normal dog and normal walk".

The existing generalist policy already completed 32/32 healthy flat-ground tests,
but it was trained across damaged bodies. This follow-up trains a separate policy
from random weights with all four legs intact, full motor strength, no scheduled
faults and level ground. Existing generalist checkpoints are preserved.

The `walk` reward profile adds healthy posture preferences: a soft 0.30 m base-height
target, stronger upright and roll/pitch angular-velocity costs, a modest nominal
joint-posture cost, a hip-spread cost, and smoother actions. These are optimization
costs, not enforced poses, phase clocks, foot trajectories or mirrored actions.
MuJoCo dynamics and torque limits are unchanged. The default `adaptive` reward is
unchanged; healthy-only training now explicitly disables initial motor weakening
as well as scheduled faults.

Start with 60 seconds of PPO using 512 batched environments, CPU MuJoCo and an MPS
learner. Continue only if evaluation indicates more training is useful, keeping
the complete checkpoint ancestry within the agreed short-run allowance.
