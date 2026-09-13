# Physics, policy and optimizer steps

The current wing_position motor body uses 5,000 Hz MuJoCo physics (0.2 ms) and
a 500 Hz policy/reward clock (2 ms): ten physics substeps per action. Four
modeled neural-core updates occur per actor call. These are separate numerical
choices; MaleCNS connectivity does not impose a 5 kHz physics requirement or
calibrate our learned latent dynamics as biological membrane dynamics.

The physics and control defaults were inherited from FlyBody's walking settings:
`_WALK_PHYSICS_TIMESTEP=2e-4`, `_WALK_CONTROL_TIMESTEP=2e-3`. Its original
aerodynamic flight setup uses `5e-5` physics (20 kHz) and `2e-4` control (5 kHz).
See the [upstream constants](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/constants.py).
The [FlyBody paper](https://www.nature.com/articles/s41586-025-09029-4) relates
small integration steps to detailed body dynamics and rapid wing flapping near
200 Hz, and explicitly separates physics steps from action sampling.

Our current wing-motion force model removes wing mass/collision/aerodynamics;
the original flight setup's exact timing requirement does not automatically
carry over. We retained the walking-rate physics to preserve the tested canonical
body. We have **not** established the minimum stable/accurate rate for this model.

The proposed smaller *critic step* means lowering the optimizer learning rate
from its current 3e-4. It does not mean changing MuJoCo's timestep. PPO's critic
fits rollout returns after collection, not once per physics substep. Moving from
5 kHz to 1 kHz physics would make the integration step **larger**, from0.2 to1 ms.

A future timing diagnostic can compare 5 kHz, 2.5 kHz and 1 kHz physics while
keeping 500 Hz actions, reward scaling and recurrent timing fixed. That gives
ten, five and two physics steps per action. Test the same frozen controller and
reference on standing, walking and hover; measure contacts, actuator response,
flight trajectories and wall time. This is a separate, explicitly versioned
physical-contract comparison, not a silent edit to accepted checkpoint physics.

MuJoCo recommends finding a timestep appropriate to the model rather than
assuming a universal rate ([performance guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#performance-tuning)).
Contact time constants also interact with timestep through the default safety
bound ([solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)).
Lowering physics frequency therefore needs a behavior check, and does not promise
a proportional end-to-end speedup while connectome inference remains unchanged.

This September13 discussion changes no physics, policy or neural clock.
