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

## Compatibility investigation

The first NVIDIA check failed before stepping: Warp 1.17's occupancy query loaded
one CCD kernel variant, then a 64-thread launch changed its symbol identity; the
next query raised a missing-symbol/metadata error. The original failed test log
is retained under ignored outputs (134.84 seconds including cold compilation).
The next attempt keeps the CCD block width at 256, matching Warp's default module
load width. This is a GPU launch configuration adjustment, not a geometry,
collision, contact-count or solver relaxation. Latest dependency pins are retained.

## Commands (on the experiment branch)

From `experiments/adaptive_locomotion` on an NVIDIA host:

```sh
uv sync --locked --extra warp
uv run --locked --extra warp pytest -q tests/test_warp_backend.py
uv run --locked --extra warp python scripts/run_physics_matrix.py

# Bounded same-parent learning pilot, after the physical checks pass.
uv run --locked --extra warp python scripts/benchmark_learner.py \
  --label warp_bridge --device cuda --physics-backend warp
```

General training supports the same independent flags: `adaptive-dog train
--physics-backend warp --device cuda ...`. Omit the physics flag for the original
CPU path. Warp requires NVIDIA CUDA; Mac viewers/evaluation retain ordinary CPU
MuJoCo. No Warp import or GPU requirement is imposed on default training.

The bridge uses real per-world reset masks and per-world actuator parameters.
Only explicit reset/forward transfers state to the device; ordinary steps upload
actuator controls and advance the GPU state. Copies back to NumPy are measured as
bridge overhead. Contact/constraint/sensor-matching overflow flags raise errors;
nonfinite state and out-of-envelope velocity stop the run. CPU evaluation remains
the authoritative task check after any GPU training.

The 256-thread workaround passed all nine topology/reset/physics tests, but its
first healthy-512 benchmark reached only **902 physics control intervals/s**,
versus **16,416/s** on CPU. Its completed reports are retained, and the in-progress
nine-body attempt was stopped before continuing the expensive matrix.
A narrower version-scoped compatibility hook now sets each generated CCD
kernel's *module default* to its already-requested 64-thread launch width. This
avoids changing its launch bounds. It uses Warp's public `set_module_options`
API, with one explicit private MJWarp builder hook guarded to the pinned versions;
review/remove it on upgrades. No solver or collision mathematics are patched.
The revised attempt must rerun all GPU checks before new throughput claims.

## Profile-driven sensor acceleration

The narrower CCD fix also passes all nine physical/reset tests (**10 tests,
95.31 seconds with cached/remaining compilation**), but healthy-512 throughput
stays near **925 physics control intervals/s**. Thus the slow result was not
explained by CCD launch width. The incomplete larger matrix was again stopped.

A three-step, 512-world CUDA-event profile attributes **186.340 / 190.284 ms
(97.9%)** of measured kernel time to `_ray`. MJWarp's default rangefinder path
scans mesh triangles; its public `rays(..., rc=...)` API also supports BVH queries.
The bridge now builds a camera-free render context solely for that spatial index,
refits it from current geometry during each sensor call, and routes only this
bridge's rangefinders through the accelerated public query. All six geom groups,
body exclusions, ray origins/directions and contact sensors remain enabled.
A second version-scoped internal dispatch hook is required because the upstream
rangefinder call does not supply a render context. No sensor values are faked.

Before claiming equivalence/speed, test **24 randomized poses per body** (seeds
9222–9224, eight worlds each, nine bodies), including body orientation and joint
variation, against CPU range distances with **0.1 mm absolute tolerance**. Test
positions span ±10 m, beyond the walking course. BVH plane bounds are finite in
MJWarp (this floor's bound extends ±24 m); this bridge is intended for the bounded
course, not arbitrary unbounded plane-ray queries. Physics plane collisions
remain unchanged. The new attempt uses `bvh_` result labels; prior results stay
available.
