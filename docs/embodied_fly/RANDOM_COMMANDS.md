# Smooth random velocity commands

The user proposed an effectively infinite source of smooth command variations.
`embodied_fly.random_velocity_commands.SmoothRandomCommands` provides the
command generator. It does not run physics or supply actions to the fly.
It is not included in velocity_recovery_dataset_01 or the ongoing recovery
imitation block, whose dataset and settings are frozen.

```python
from embodied_fly.random_velocity_commands import SmoothRandomCommands
source = SmoothRandomCommands(seed=141101)
command = source.command(simulated_seconds)
# forward / left / up in cm/s, yaw in rad/s
```

Default bounds match the current exercise: each translation component at most
1.5 cm/s, yaw at most 4.5 rad/s. Each two-second segment has a one-second
quintic transition and a one-second hold. Every fifth segment goes to zero
velocity for braking/hover. Each component also has a chance of zero, so isolated
and combined requests occur. Targets are reproducible by seed and segment
index, with a bounded cache; generating a later segment does not require storing
or replaying every earlier command. Tests cover bounds, reproducibility, zero
holds and continuity of commands and their first two derivatives at joins.

This produces unlimited command sequences, not free physical experience. A
future collector still needs to simulate the teacher and retain labels/states.
Random vertical velocity integrates into displacement, so an actual collector
needs an explicit altitude envelope and smooth command-boundary handling. That
physical validation has not been done for this generator. Keep full-skill
exercises for coverage/retention tests, use different random seeds for validation,
and archive the finite data/seed prefix actually used in each timed training run.

The practical follow-up is to add validated random episodes as a separately
identified source, retaining complete coverage exercises and student-state PID
recoveries. Do not silently replace an existing timed run's dataset.
