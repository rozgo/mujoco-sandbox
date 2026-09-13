# Physical PPO after PID imitation

User-approved continuation of pid_imitation_01. Evaluate the final handoff actor
first. If it collapses, evaluate the saved end-of-full-teaching checkpoint too;
select by count of complete airborne review starts, then nominal failure time
if neither survives, or nominal position RMS among survivors. Record selection
explicitly as development/model selection, not held-out evaluation. The goal is
one continuing actor, with no teacher in this PPO stage or deployed controller.

Run300 requested training seconds,64 hover worlds,16 CPU MuJoCo/mjbatch threads,
RTX4090 neural work. Same399 inputs/78 outputs, fixed MaleCNS graph and learned
cell/interface architecture. Accepted instantaneous1 kHz physics,500 Hz actions.
Bounded hover reward retained from pilot06: vertical-speed scale2 cm/s, all
other terms identical. No imitation loss or runtime wing oscillator.

Actor LR1e-7; critic LR1e-4; std/floor.001; horizon512; sequence128; two epochs;
gamma.999; GAE.995; KL limit.03; entropy0; five-second episodes. New PPO Adam,
new critic/critic Adam, four critic-only fitting rollouts with frozen actor,
all included in measured training time. Seed120402. Actor weights come from
the selected imitation checkpoint; this is not uninterrupted PPO optimizer history.

Compare all three frozen ten-second starts against accepted PID with the same
physical model and gates. Preserve failures and repeated height ripple separately
from startup/altitude bias. Render/open the comparison. No next motor task or
utility stage is introduced. Record exact source, ancestry, hashes and all times.
