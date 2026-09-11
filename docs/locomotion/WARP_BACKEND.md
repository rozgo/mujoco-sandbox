# Experimental GPU physics backend

Started 2026-09-11 17:32:50 UTC (first retained clock checkpoint). The user asked
for a separate MuJoCo Warp path to accelerate physical simulation on the RTX 4090.
Work stays on `experiment/adaptive-dog-gpu-comparison`; official v1 stays frozen.

## Plan declared before measurement

- Keep `mjbatch` CPU physics as the default and native Mac evaluation/viewing path.
- Add optional `--physics-backend warp`, independently of the PyTorch learner
  device. Pin current releases MuJoCo Warp 3.13.0 and Warp 1.17.0 in the uv extra.
- Preserve all nine real body models, their contact geometry, sensors, joint and
  torque limits, solver settings, and 2 ms training / 20 ms control clocks. Batch
  each topology separately; do not replace missing limbs with invisible parts.
- First implement a compatibility bridge: physics and CUDA graphs on GPU, existing
  NumPy observations/rewards/reference matching on CPU. Disclose host transfers;
  this is not an entirely GPU-resident RL loop.
- Validate deterministic reset and short fixed-action trajectories against CPU,
  contact sensors and actuator limits, finite state and contact/constraint capacity.
  Predeclared short-step tolerances: max qpos error 0.002, qvel error 0.1 over ten
  2 ms steps from identical initialized states and controls. Long contact-rich
  trajectories may diverge; judge those by task and per-leg gait metrics.
- Measure cold setup/compilation separately from synchronized warm stepping.
  Compare 512 and 4096 environments, healthy and all nine bodies, CPU and Warp,
  three 3-second repeats per configuration. No physics-only speed claim may be
  presented as end-to-end RL throughput.
- If correctness passes and the bridge is usable, run one bounded 90-second PPO
  continuation with the previous frozen parent, motion bank, seed and recipe,
  followed by eight CPU evaluation trials per body. Preserve the final checkpoint
  and all failed gates. Do not tune or promote from this pilot. If bridge overhead
  dominates, document it and stop before expensive training.
- Leave unrelated GPU workloads running. Record load and limit interpretation.

## Sources

[MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)
explains GPU batching, graph capture, capacity checks and supported features.
[Official source](https://github.com/google-deepmind/mujoco_warp) and the installed
3.13.0 API are the implementation references. Release versions were checked on
[PyPI](https://pypi.org/project/mujoco-warp/) before pinning the optional stack.
