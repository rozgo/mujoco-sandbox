# Microscale unit-rescaling study

User request, October 5, 2026: test consistent internal unit rescaling for the
40 µm flexible thread and document special requirements, pros and cons. Source
parameters and reported measurements remain SI; the engine may use converted
units. The existing static workcell remains available unchanged.

## Acceptance targets, fixed before the main evaluation

- No numerical warnings, nonfinite state, or silent resets in completed cases.
- Exact physical mass and dimensions after unit conversion; no inertia floors
  or extra armature added to get the model to compile.
- Small-deflection cantilever: finest mesh within 3% of `F L³ / (3 EI)`;
  32-to-64-segment tip change under 3% of that reference.
- Same physical trajectory across valid unit systems to within 0.01 µm.
- Timestep refinement: maximum tip difference below 1 µm at matched timestamps.
- Contact: peak support penetration below 2 µm, settled penetration below
  0.2 µm. Both are numerical test gates, not surgery safety limits.
- Inextensible segment lengths remain within 1 nm of their specified lengths.

**Result: rescaling clears the inertia limit, but the contact model is not
accepted.** Bending passes the preliminary checks below. Rebound fails timestep
convergence even at 78.125 ns; increasing compute alone has not resolved it.
Integration into the robot and RL are held at this physics gate.
The follow-up [solver decision and diagram](SOLVER_DECISION.md) document the
recommended independent benchmark and Isaac Sim 6.1 integration option.

The [measured results](RESCALING_RESULTS.json) retain the complete final suite,
and [pilot records](RESCALING_PILOTS.json) retain all 36 exploratory cases,
including failures. Raw states/MJCF remain in ignored
`outputs/neural_insertion/rescaling/`. Review the
[accuracy plots](../../previews/neural_insertion/rescaling/quality_review.png)
and [recorded contact pose](../../previews/neural_insertion/rescaling/contact_diagnostic.png).

## Measured results — standard MuJoCo 3.12.0, Apple Silicon CPU

| Check | Measurement | Outcome |
| --- | --- | --- |
| Original m–kg–s cable | Minimum-inertia compilation failure | Cannot use this articulated cable directly at SI scale |
| Same loaded bending in mm–kg–s, mm–g–s, cm–g–s | Maximum matched tip difference 5.2e−12 µm | Pass 0.01 µm gate |
| 10 mm cantilever, 10 nN, 64 segments, 2 s | 258.892 µm deflection vs 265.258 µm reference | 2.400% error; pass preliminary 3% gate |
| Bending mesh, 32 → 64 segments | 6.110 µm change, 2.303% of reference | Pass preliminary 3% gate |
| Frictional contact, finest two timesteps | Peak penetration 1.739 µm | Pass 2 µm numerical gate |
| Contact timestep 0.15625 → 0.078125 µs | Maximum matched tip/shape separation 34.979 µm | **Fail 1 µm gate** |
| Physical mass / fixed segment length | Correct nominal mass; maximum length error 1.19e−8 nm | Pass for final fixtures |
| Completed rescaled cases | No warnings or nonfinite state; no resets | Numerical completion only |

The independent cantilever reference is the small-deflection Euler–Bernoulli
expression `delta = F L³/(3 EI)`. The fixed first segment discretizes the clamp;
the mesh error is visible and still matters. Passing this preliminary test does
not establish 1 µm placement accuracy. The contact fixture is a 44 mm rod clamped
at one end, released from horizontal under gravity onto a support 6 mm below.
It has 32 segments and no added bending damping. Both final runs cover the same
60 ms, sampled every 0.1 ms; penetration is inspected every physics step.

The final-quarter contact window still includes rebound. It is **not a settled
contact measurement**; the 0.2 µm settled gate remains unvalidated. Contact spatial
refinement, torsion accuracy and interaction with a tool or tissue are also
unvalidated. These are deterministic benchmarks, not robustness trials.

Additional diagnostics did not cure contact convergence: RK4, zero sliding
friction, 2 ms bending damping, and larger mass-unit multipliers. Removing friction
was a diagnostic only. At 1.25 µs, switching from mm–mg–s to mm–µg–s changes the
matched rebound trajectory by 13.249 µm despite identical physical parameters.
Unit agreement is therefore established for the bending fixture only. This
sensitivity warrants contact/rod solver investigation and an independent reference;
it does not prove that every MuJoCo microscale model is infeasible.
The [contact-isolation follow-up](CONTACT_ISOLATION.md) checks compiled physical
equivalence first, then tests mass and length unit changes separately during
contact. Agreement before stepping does not imply agreement during impact.

The final nine-case suite took **137.04 s elapsed**, including **136.80 s summed
trial wall time**, for **7.02 s aggregate simulation time** (one case fails at
compilation). Trial wall time includes Python telemetry and model compilation.
The finest contact case alone took 82.43 s for 0.06 s simulated, roughly 1,374×
slower than real time in this harness. This is not an optimized solver benchmark
or an RL-throughput estimate. No training ran.

## Reproduce

```sh
# Use a fresh output path for each run. Overall exit 1 currently means the
# recorded contact-convergence gate failed; inspect manifest.json.
uv run --locked python -m sixlegs.neural_insertion.benchmarks \
  --output outputs/neural_insertion/rescaling/recheck_v1

# Fast conversion-only check; --suite bending or contact selects other groups.
uv run --locked python -m sixlegs.neural_insertion.benchmarks \
  --suite units --output outputs/neural_insertion/rescaling/units_recheck_v1

uv run --locked pytest -q tests/test_neural_insertion.py tests/test_neural_rescaling.py

# Plot and replay saved states; matplotlib comes from the existing wind extra.
uv run --locked --extra wind python scripts/report_neural_rescaling.py \
  outputs/neural_insertion/rescaling/recheck_v1 \
  previews/neural_insertion/rescaling/recheck_v1
```

The manifest records engine version, source commit plus hashes of the uncommitted
implementation, configuration, units, scene hashes and trajectory hashes. Cases
save qpos/qvel in engine units and tip/vertices in SI. Existing result directories
are preserved. The earlier untested point-mass/native draft is archived under
ignored `build/neural_insertion/archived_si_draft/`; it is not the active model.

Material assumptions: radius 20 µm, density 1400 kg/m³, Young modulus 100 MPa,
Poisson ratio 0.45, circular isotropic section. These are illustrative values,
selected to exercise a compliant slender thread; no physical thread was calibrated.
Contact uses sliding friction 0.3, impedance 0.9999, a 10 µs time constant and
unit damping ratio. Newton uses 100 iterations and tolerance 1e−12; observed
iteration counts in the finest pair remain below the cap (11 and 14).

## Unit contract

Let `L` be engine length units/metre and `M` engine mass units/kilogram. Keep
seconds and radians unchanged. The preferred candidate is **mm–g–s**:
`L=1000`, `M=1000`. Convert at the model/input boundary and invert the conversion
for observations, telemetry, measurements and exported trajectories.

| Quantity | SI → engine multiplier | mm–g–s multiplier |
| --- | --- | --- |
| Position, radius, travel, linear velocity/acceleration | L | 1,000 |
| Mass | M | 1,000 |
| Rotational inertia, torque, energy | M L² | 1,000,000,000 |
| Force | M L | 1,000,000 |
| Density | M/L³ | 0.000001 |
| Young/shear modulus, pressure | M/L | 1 |
| Bending rigidity EI | M L³ | 1,000,000,000,000 |
| Linear spring/servo stiffness and linear damping | M | 1,000 |
| Angular stiffness and angular damping | M L² | 1,000,000,000 |
| Slide-joint armature | M | 1,000 |
| Angular-joint armature | M L² | 1,000,000,000 |
| Sliding friction coefficient | 1 | 1 |
| Torsional/rolling contact-friction coefficients | L | 1,000 |
| Angles, quaternions, unit axes, damping ratios | 1 | 1 |
| Time, timestep, positive solref time constant | 1 | 1 |

Examples: 40 µm diameter becomes **0.04 mm**; gravity becomes **9810 mm/s²**;
1400 kg/m³ becomes **0.0014 g/mm³**; 10 nN becomes **0.01** force units. A 44 mm
thread still has physical mass **7.7408843e−8 kg**. The engine numbers change;
the physical object and passage of time do not.

Source of the unit convention: [MuJoCo units documentation](https://mujoco.readthedocs.io/en/stable/overview.html#units-are-unspecified).

## Details that must be handled explicitly

1. **Override defaults.** Gravity and density have SI-like defaults. Geometry,
   mesh vertices, body/inertial offsets, sites, cameras, joint travel and actuator
   bounds must all agree with the engine units. Never scale only the thread in an
   otherwise SI robot scene. Keep a unit-system identifier in scene/run manifests.
2. **Scale the cable material, not just its geometry.** The built-in
   `mujoco.elasticity.cable` plugin takes Young/shear moduli, computes circular
   area moments from radius, and supplies bending/twist forces. It does not need
   an extra external bending-force implementation. A native process must load
   MuJoCo's elasticity plugin library; the Python package loads its bundled plugins.
3. **Set generated cable contacts directly.** Composite cable geoms have internal
   defaults. The pilot accidentally combined a cable time constant of 20 ms with
   the support's 0.6 ms setting, producing a 10.3 ms contact time constant. Inspect
   compiled `geom_solref/solimp` and actual contact parameters, not just the parent
   XML defaults. Set both participating surfaces consistently or use an explicit
   pair/priority. Contact friction and solver settings also need explicit values.
4. **Scale distance tolerances.** Contact margins, gaps and the contact `solimp`
   width are lengths. Convex-collision tolerances also need a declared physical
   scale when we add mesh/tool contacts. The current plane–capsule test uses an
   analytical collision pair. Dimensionless impedance values stay unchanged.
   Solver stopping tolerance is a numerical parameter: record it and verify unit
   equivalence, rather than assigning it an assumed force-unit conversion.
5. **Keep mass accounting exact.** Overlapping capsule endcaps must not add
   material to every segment. Each segment gets explicit mass `rho*pi*r²*ds`.
   Inertia follows that capsule mass distribution; no `boundinertia`, extra
   armature or hidden mass multiplier is used. Segment discretization still
   approximates the continuous rod and must be refined.
6. **Treat spatial and temporal resolution separately.** A 40 µm diameter does
   not imply 40 µm resolution along a long segment. The analytical bending test
   uses 16/32/64 segments. Generated composite vertices can incur small floating
   point rounding; a 44 mm thread divided into 32 segments gives exactly
   representable 1.375 mm spacing in the selected unit system.
   The 10 mm/16-segment unit-equivalence fixture is exact in all three tested
   unit systems. In contrast, changing the 44 mm/32-segment fixture to cm–g–s
   introduces generated-vertex rounding (up to about 4.1 ppm in inferred inertia).
   Compare compiled geometry, not only nominal XML dimensions; explicitly generated
   high-precision bodies are a candidate for subsequent arbitrary-length rods.
7. **Small contact time constants require small real timesteps.** `refsafe`
   remains enabled. A positive `solref` time constant must be at least twice the
   timestep; rescaling lengths does not change this time relationship. Report
   penetration at every substep and compare entire trajectories at matched times.
8. **Do not confuse damping with a numerical free lunch.** The bending benchmark
   includes a declared Kelvin-type rotational damping coefficient
   `c_theta = relaxation_time * EI/ds`, with a 2 ms relaxation assumption.
   This changes physical transients, so an additional zero-bending-damping contact
   suite tests the elastic limit. Material damping needs measurement before we
   call the model calibrated. Contact damping is specified separately.
9. **Integrators must actually be exercised.** The split `mj_step1/mj_step2` path
   runs the implicit integrator used here, but does not implement RK4. Passive
   RK4 checks call `mj_step`; a loaded RK4 experiment would need forces recomputed
   at its internal stages. The benchmark rejects that unsupported combination.
10. **Keep force application and reporting consistent.** Apply loads at the
    physical endpoint with `mj_applyFT`, after current kinematics are computed.
    Convert forces and torques separately. Reset/replay state assignment is
    separate from live stepping. Learned observations/rewards should use SI or
    explicitly normalized SI, so changing engine units cannot change the task.

Solver guidance: [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).
Cable model: [MuJoCo elasticity plugin](https://github.com/google-deepmind/mujoco/blob/3.12.0/plugin/elasticity/README.md).

## Pros and cons

**Advantages:** preserves physical dimensions and mass; moves tiny inertias away
from hard numerical floors; permits the built-in elastic cable model with bending,
twist and collision; keeps mechanics in the same engine as the robot; allows
independent unit systems to expose conversion mistakes.

**Costs and limits:** every dimensional interface needs conversion; rescaling
preserves the robot/thread mass ratio and cannot by itself cure poor conditioning;
fast impacts may need sub-microsecond timesteps; segment refinement increases
state size and solver cost; physical damping and contact assumptions still need
calibration; a converged numerical model can still describe the wrong material.

The cable is an **inextensible circular rod**. It bends and twists, while segment
length is constrained by the joint tree. Axial stretch, plasticity, fracture,
adhesion, fluid effects and tissue puncture are absent. The inextensible assumption
must be checked against expected tension using `strain = tension/(E*A)`; use an
extensible model if the omitted elongation is material to placement accuracy.

User steering: simulation quality takes priority over performance. Failed accuracy
or convergence gates remain failures; runtime alone is not a reason to loosen them.

Next investigation: isolate the contact/elasticity coupling against an independent
rod/contact reference, then repeat temporal and spatial refinement with measured
material/contact parameters. Consider an implicit rod or FEM subsystem coupled to
MuJoCo if the current formulation cannot meet the accuracy gates. Higher resolution
and slower execution are acceptable; increased mass/radius, invisible supports,
disabled task collisions or unreported damping are not substitutes for accuracy.
