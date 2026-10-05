# Surgical thread physics: next solver decision

October 5, 2026. The [MuJoCo DER port and acceptance study](DER_VALIDATION.md) is
now implemented. A separate plugin adds a corrected force-projection variant;
the installed engine is unchanged. Contact accuracy remains unresolved, and no
robot integration or training has started. Other solver routes remain proposals.
The subsequent [contact-isolation study](CONTACT_ISOLATION.md) tests failures
without rod elasticity and explicitly audits rescaling. It explains each control,
its interpretation and the limits of the result.

## Recommendation

**Try the published discrete elastic rod (DER) model inside MuJoCo first.**
The user's follow-up changes the earlier Newton-first proposal to this order:

| Priority | Candidate | Decision point |
| --- | --- | --- |
| 1 — MuJoCo | Adapt `qj25/adapteddlo_muj` to our pinned engine and benchmark fixture | Retain if rod mechanics, contact and release pass refinement checks |
| 2 — Isaac Sim | Evaluate Newton VBD rod/rigid contact in one solver, then verify the Isaac bridge | Test the solver standalone before paying the platform integration cost |
| 3 — Combination | MuJoCo rigid robot plus a dedicated rod solver, with Isaac as an optional host; consider Newton coupling or DeformX | Accept only after interface timestep/iteration refinement and impulse/work checks |
| 4 — Other solvers | SOFA Cosserat/BeamAdapter, then an implicit DER implementation such as DisMech; PyElastica as another reference candidate | Validate the same fixture before a robot port or custom solver |

These are investigation priorities, not accuracy rankings. DeformX already is a
combination of Isaac/PhysX and an external rod engine; it belongs to priority 3.
An independent analytical or numerical reference can be used at any stage without
migrating the robot. Preserve the MuJoCo scene and failed cable baseline. Quality
remains the deciding factor, not speed.

If existing solvers pass, use their implementation. If a specific MuJoCo defect
is isolated, make a narrow patch with a regression test. If a solver capability
is missing, a dedicated rod/contact module is preferable to broad engine changes.
Changing packaging or tuning constants does not by itself solve convergence.

### MuJoCo DER source audit

Chen, Bretl and Pham's model maps DER elastic forces into MuJoCo joint torques
and retains MuJoCo contact handling. It still uses a capsule/ball-joint chain;
the important change is the elastic formulation. Its assumptions include
inextensibility, negligible shear and quasistatic material-frame twist. Reported
buckling and real-cable comparisons support testing it, but do not establish our
microscale contact accuracy. Quasistatic twist must be justified for our release
timescale; removing torsional dynamics is a model assumption, not automatically
an accuracy improvement. [Paper, revision 3](https://arxiv.org/html/2310.00911v3).

Inspected upstream revision `71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad`.
Its C++ plugin instructions target **MuJoCo 3.3.2**, require Eigen and source
changes for a custom composite/XML type. It is not an already compatible 3.12.0
plugin. First investigate generating the articulated model explicitly and porting
the force implementation through supported APIs; retain upstream license notices.
Reproduce an upstream benchmark before interpreting microscale differences.
[Repository](https://github.com/qj25/adapteddlo_muj/tree/71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad),
[build instructions](https://github.com/qj25/adapteddlo_muj/blob/71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad/wire_plugin/README.md),
[model-builder changes](https://github.com/qj25/adapteddlo_muj/blob/71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad/wire_plugin/add2source.md).

Two claims from the supplied research remain unestablished for our experiment:

- **The original contact result did not isolate its cause.** The 35 µm disagreement cannot
  distinguish contact regularization, elastic integration, spatial discretization
  or their interaction. A passing small-deflection cantilever does not validate
  large-deformation rod dynamics. Compare free relaxation, slow contact and rebound
  while changing one numerical choice at a time. The original run already used
  Newton, not PGS, with a 100-iteration cap; iteration count alone is not a diagnosis.
  The later isolation controls demonstrate failed refinement without elasticity
  and unit sensitivity in the articulated chain; a specific engine defect remains
  unestablished.
- **The old inertia report is not an established current defect.** Issue #2208 is
  closed. Inspected 3.12.0 source selects circular `I=πr⁴/4`, `J=πr⁴/2` for capsule
  geoms. Check compiled geom selection and measured stiffness before blaming or
  patching that code. [Issue](https://github.com/google-deepmind/mujoco/issues/2208),
  [pinned source](https://github.com/google-deepmind/mujoco/blob/3.12.0/plugin/elasticity/cable.cc).

None of the sources checked here establishes a 1 µm contact/rebound result for
our geometry. This is a bounded literature finding, not a claim that no such
publication exists. The 1 µm gate remains a numerical refinement requirement;
physical accuracy also needs measured bending/torsional stiffness and contact data.

## What we established

The rescaled 40 µm cable compiles with its physical mass and radius. Bending is
within 2.4% of the small-deflection reference. Peak support penetration is
1.739 µm. However, halving the timestep from 156.25 ns to 78.125 ns changes the
rebound trajectory by **34.979 µm**, failing our **1 µm** convergence gate.
This is disagreement between two numerical solutions, not measured error against
a real thread. It does not identify the root cause or disqualify every MuJoCo
deformable formulation. [Full study](RESCALING.md), [measured data](RESCALING_RESULTS.json).

![Measured bending and contact convergence](../../previews/neural_insertion/rescaling/quality_review.png)

The tests used an articulated cable with the built-in elasticity plugin. Its
passive-force computation adds bending/twist torques to the system. MuJoCo 3.12.0's
plugin interface has state, compute, advance and visualization hooks, but no
general callback for replacing the global contact solver or supplying arbitrary
elastic stiffness to that solver. A replacement force callback therefore does
not automatically provide fully implicit rod/contact integration.
[Plugin interface](https://github.com/google-deepmind/mujoco/blob/3.12.0/include/mujoco/mjplugin.h),
[cable implementation](https://github.com/google-deepmind/mujoco/blob/3.12.0/plugin/elasticity/cable.cc).

The ordinary `implicitfast` path is implicit in velocity; its name does not imply
that all plugin elasticity is solved implicitly in position. Version 3.12.0 also
contains a separate implicit flex path gated on CG, a non-elliptic cone, sleep
disabled, and eligible flex stiffness/contact. Our articulated cable with Newton
and elliptic contacts did not exercise that path. Merely selecting CG would not
turn the cable into an eligible flex. A volumetric flex is an untested alternative,
with its own mesh and constitutive requirements.
[Integrator description](https://mujoco.readthedocs.io/en/3.12.0/computation/index.html#numerical-integration),
[pinned flex gate](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_forward.c#L1590-L1645).

These source findings explain why more than force tuning may be needed. They are
**not proof of the cause** of the recorded rebound sensitivity. No claim is made
that a particular missing Jacobian caused the failure.

## Options

| Route | What it offers | Main cost / uncertainty | Decision |
| --- | --- | --- | --- |
| Published DER inside MuJoCo | Rod elastic energy with existing rigid/contact engine | 3.3.2 code needs porting; quasistatic twist; same contact formulation | First candidate |
| Narrow MuJoCo patch | Retains one dynamics engine; fixes a demonstrated defect | Requires a minimal reproducer and correct reference; maintenance burden | Only after isolating a specific defect |
| New passive-force plugin | Custom material, anisotropy, damping and observations | Retains native integration/contact limitations unless separately addressed | Useful packaging, insufficient evidence of a fix |
| Dedicated rod/contact module coupled to MuJoCo | Own integration, material state and contact treatment; keeps robot | Coupling can create energy or force errors; substantial validation work | Preferred custom route if existing models fail |
| Native MuJoCo flex volume | Existing continuum path and eligible implicit treatment | Untested at this geometry; resolving a thin, long volume is expensive | Lower priority for the thread |
| Isaac Sim / PhysX deformables | GPU volumetric FEM and robot/sensor environment | Microscale accuracy unproven; thin-thread mesh conditioning; platform migration | Secondary candidate, especially for tissue |
| Newton VBD rods | Explicit rod stretch, shear, bend and twist support | Microscale convergence and Isaac integration/version compatibility unproven | Second priority after MuJoCo DER |
| SOFA BeamAdapter | FEM beam formulation and collision examples | Must establish its own refinement and physical validity | Independent comparison first |

SOFA BeamAdapter documents corotational beam elements for cables, threads and
flexible needles; its collision examples are relevant starting points. The
separate SOFA Cosserat plugin is another rod formulation if BeamAdapter cannot
represent the required constitutive model. Neither is already validated for our
40 µm case. [BeamAdapter](https://sofa-framework.github.io/BeamAdapter/),
[SOFA Cosserat](https://github.com/SofaDefrost/Cosserat).

DisMech is a further candidate because it combines DER mechanics with implicit
contact treatment. Its existence does not validate our geometry or material.
[DisMech paper](https://arxiv.org/abs/2311.18126),
[rod implementation](https://github.com/StructuresComp/dismech-rods).

## Would Isaac Sim meet the resolution requirement?

**Possibly, but its documented features do not establish our accuracy target.**
PhysX deformable volumes use tetrahedral simulation/collision meshes and a
neo-Hookean material model; the deformable solver is XPBD. Deformables require
GPU simulation. Those are useful capabilities, not a guarantee of sub-micrometre
forces or placement accuracy.
[PhysX deformable volume](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/DeformableVolume.html),
[solver API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/_api_build/classPxDeformableBody.html),
[GPU requirement](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/deformables.html).

The 44 mm thread has a length/diameter ratio of **1,100**. A fine-looking surface
mesh is not evidence of adequate simulation mesh resolution. We must test axial
and cross-sectional refinement, solver iterations, timestep, actual arithmetic
precision, contact offsets and physical unit scaling. Contact/rest offsets can
change contact detection or effective separation and cannot be left at unrelated
scene defaults. Physical material parameters, mass and radius remain fixed.
[Isaac contact settings](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html).

Isaac Sim 6.1 documents experimental Newton integration and an engine interface.
The current Newton VBD API explicitly supports rod joints and their stretch,
shear, bending and twist response. This makes **Newton VBD more directly matched
to our thread geometry**—an engineering judgment, not a measured accuracy ranking.
Pin the actual Newton build and verify that the selected Isaac Sim extension
exposes the required solver/features. Current standalone Newton documentation is
not proof of what a bundled Isaac release supports. A Newton MuJoCo backend is
also different from Newton VBD; choose the solver explicitly.
[Isaac 6.1 Newton backend](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/newton_physics.html),
[VBD rod API](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverVBD.html).

Standalone VBD also documents a unified rigid/rod solve; separate MuJoCo/VBD
coupling is an option, not a requirement of all VBD use. This is why a small
single-solver VBD fixture precedes the combination route in the priority order.
[VBD guide](https://newton-physics.github.io/newton/latest/solvers/vbd.html).

The 6.1 bridge documentation demonstrates Newton/MuJoCo, does not document VBD
selection, and says additional solver-specific scene classes are under development.
It also requires `metersPerUnit=1.0`; it does not convert non-unit USD stages on
import. Our mm–g–s convention cannot be carried over simply by changing USD
metadata. Any internal rescaling needs explicit backend conversions and fresh
invariance tests. Keep standalone Newton development isolated from Isaac's tested
dependency versions. These are bridge constraints, not limits on all standalone
Newton formulations. [Bridge details](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/newton_physics.html).

Full Isaac Sim requires a supported Linux/Windows NVIDIA RTX machine; it does
not provide a native Apple Silicon route for this workstation. Check the exact
release's compatibility requirements before provisioning. Standalone Newton
benchmarking is a separate deployment decision. PufferLib 5.0 remains our learning
target; an Isaac/soft-solver environment adapter would be custom work, and a move
to Isaac Sim does not imply adopting Isaac Lab or an existing Puffer integration.
[Isaac installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html),
[hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).

## Proposed boundary and validation gate

[Open the interactive architecture diagram](../../previews/neural_insertion/solver_plan_v1/physics_architecture.html).
It shows a **proposed Isaac Sim 6.1 custom backend**, not implemented runtime code.
Preserved as a possible integration route after the MuJoCo-first follow-up; the
priority table above is the current experiment order.

![Proposed Isaac Sim 6.1 surgical backend](../../previews/neural_insertion/solver_plan_v1/physics_architecture.png)

Diagram validation: 9/9 showcase checks, browser checks passed, light/dark renders
visually inspected. [Portable review receipt](../../previews/neural_insertion/solver_plan_v1/REVIEW.json).

The supplied [6.1 integration page](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)
defines the needed adapters: USD ingestion, simulation/lifecycle callbacks,
Fabric updates and tensor views. Only one engine is active at a time; switching
reloads USD and invalidates views. Newton's current tensor coverage is rigid
bodies, articulations and rigid contacts, whereas PhysX includes deformables.
A composite robot/rod backend must own its coupled stepping and expose any
additional soft-body state. This is an integration route, not a numerical solver.

A passing unified rod/robot solver could avoid cross-engine coupling. If separate
MuJoCo and rod solvers are needed, place both behind one adapter and validate their
coupling. In either case, use Isaac for authoring, rendering and sensor integration
only after the standalone physical benchmark succeeds.

If coupling is needed:

- MuJoCo owns rigid robot state and bounded actuators. The soft solver owns
  thread state, its contacts, and later tissue deformation/puncture/retention.
- Exchange poses, linear/angular velocities, time and reaction wrenches through
  explicit SI/frame conversions. Apply equal/opposite forces at the same physical
  points. Assign each contact to exactly one solver to avoid double forces.
- Substep and iterate coupled trial states until position/force residuals meet
  declared tolerances; check work/energy balance. Accept one synchronized state.
  Advancing each solver once with stale forces is not assumed accurate.
- Refine the exchange interval separately from each internal timestep. Check
  linear/angular impulse balance including support reactions, and account for
  external work and physical dissipation. After release, test free relaxation
  for spurious acceleration or energy growth; a bent rod is allowed to move.
- Preserve all rod/material/contact state through reset, copy, replay and batched
  RL environments. A visual mesh driven by those states is observer output.
- A calibrated rod should represent bending/twist and the needed axial/shear
  compliance, with section-dependent stiffness. Puncture is a separate tissue
  interaction law; generic soft contact does not automatically implement it.

Newton documents lagged/staggered proxy and iterative ADMM coupling. DeformX
documents fine rod substeps with predicted rigid motion and accumulated reaction
impulses; its paper explicitly identifies approximate two-way coupling as a
limitation. Neither approach establishes our microscale gate. DeformX's current
README targets Isaac 4.5/5.x, so 6.1 compatibility must also be checked rather
than assumed. Its rigid-motion predictor uses semi-implicit Euler; this is not
evidence of a fully implicit rod/contact solve.
[Newton coupling](https://newton-physics.github.io/newton/1.5.0/concepts/coupling.html),
[DeformX paper](https://arxiv.org/html/2606.22116v1),
[DeformX setup](https://github.com/DeformX/DeformX#environment-setup).

## Next experiment, before a robot port or custom implementation

Steps 1–4 have now been partially exercised by the [DER study](DER_VALIDATION.md):
force/lifecycle checks and preliminary bending/relaxation pass, but contact
refinement fails. Full upstream buckling reproduction and physical tool release
remain unvalidated. The sequence below defines remaining gates, not completed work.

1. Pin and port the published MuJoCo DER implementation in a separate benchmark
   variant on `main`. Preserve the approved scene and failed native-cable runs.
   Reproduce upstream bending/twist buckling behavior before microscale evaluation.
   Use supported model/force APIs where possible; compiler integration changes
   must not silently change the dynamics solver.
2. Compile and inspect the 40 µm × 44 mm DER fixture, then compare it with the
   native cable using matched mass, section stiffness, contact law and conditions.
   Apply only one elastic model per rod; do not add DER forces atop cable torques.
3. Run the cantilever, a large-bend/twist case and free relaxation, followed by
   slow load/unload contact and the original frictional drop/rebound. Refine time,
   segment count, solver tolerance/iterations and units separately. Record force,
   energy and contact-event timing alongside matched-time shape error.
4. Retain existing gates: less than 1 µm trajectory change under refinement,
   less than 2 µm peak penetration, less than 0.2 µm genuinely settled penetration,
   and unit invariance. Retain the preliminary 3% bending check; add matched force,
   energy and constitutive validation before acceptance for manipulation. Evidence
   must show a refinement trend, not only an accidentally close timestep pair.
5. If DER still fails, use the diagnostics to decide whether a narrow MuJoCo fix
   is justified. Otherwise move to Newton VBD for the Isaac route, then validated
   coupling, then the standalone alternatives in the priority table. Use an
   independently refined reference when needed, with contact-law differences
   disclosed. Avoid an open-ended engine fork.
6. With a passing candidate, test physical tool/thread hold and release, then a
   bounded robot approach. Start PufferLib alignment/tracking RL at that point;
   tissue insertion and richer thread handling develop progressively.

If these candidates fail too, save that outcome and reassess the model/task.
Do not broaden a fork or hide error with altered mass, radius or unreported damping.

Sources checked October 5, 2026. MuJoCo claims above use installed version 3.12.0
and pinned source; moving `latest` Isaac/Newton documentation is discovery evidence,
not a dependency lock. No cross-engine accuracy or performance result is claimed.
