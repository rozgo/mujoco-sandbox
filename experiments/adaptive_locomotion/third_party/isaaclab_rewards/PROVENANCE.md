# Isaac Lab reward attribution

Reward formulas in `adaptive_locomotion/gait_balance.py` adapt Isaac Lab's Spot `air_time_variance_penalty` and `base_motion_penalty` to NumPy/MuJoCo batching. The timing counters and healthy-body gating are local implementations. No Isaac runtime dependency is added.

- Source revision: `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`
- [Original rewards](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py)
- Source copyright: Copyright (c) 2022-2026, The Isaac Lab Project Developers.
- License: [BSD-3-Clause](LICENSE), retained from that revision.
- Accessed September 10, 2026.

`adaptive_locomotion/foot_clearance.py` also uses the speed-weighted height-error
idea from `foot_clearance_reward`. Local changes use terminal sphere bottoms,
a one-sided 3 cm clearance deficit normalized to a bounded cost, finite-difference
world velocity, and masks excluding healthy bodies, stumps and absent legs.
It does not require Isaac Lab and introduces no contact-phase schedule.
