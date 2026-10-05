# Why isolate contact?

Our intended thread is flexible. Its model bends and twists. A rigid segment is
only a diagnostic control: removing elasticity lets us ask whether the same
numerical problem occurs without bending and twisting forces. It is not a
replacement thread or a proposal to make the surgical task rigid.

**Result: the minimum contact guarantees still fail.** Both timestep sensitivity
without elasticity and unit sensitivity in articulated contact are demonstrated.
The physical conversions pass inspection, but rescaling cannot yet be treated as
numerically harmless during contact. Robot integration and RL remain on hold.
The subsequent [focused audit](CONTACT_AUDIT.md) checks same-state forces and an
independent rigid reference. It finds consistent force calculations, fixed-step
impact errors and amplification of tiny state differences at contact entry.

The corrected DER model passes force/energy consistency and preliminary free
relaxation checks, but changing the timestep still changes its contact trajectory
by tens of micrometres. Before changing the rod model or patching MuJoCo, we need
to learn which ingredients are necessary for that failure. These are numerical
consistency tests, not measurements against a physical thread.

## What each test tells us

| Test | Kept | Removed or changed | Why run it? |
| --- | --- | --- | --- |
| Straight segment drop | One actual 1.375 mm segment's 40 µm diameter, mass, inertia and support contact law | Other segments, joints, bending and twisting | Check the simplest normal impact at the same cross-section scale. |
| Angled/sliding segment | Same section and contact law | Start at 15° with 0.05 m/s lateral velocity | Exercise friction and rotation that a perfectly symmetric drop can miss. |
| Chain without elasticity | All 32 segments, joints, inertia, collision geometry and pre-impact positions/velocities | Elastic plugin; no joint damping | Ask whether an articulated chain can fail without any rod elastic forces. |
| Flexible DER chain | Same complete pre-impact chain and contact settings | Elastic forces enabled | Compare the actual candidate model against the controls. |

All motion after initialization comes from MuJoCo stepping. No forces or state
resets are used to make an impact succeed. The two chain cases start from the
same saved 30 ms checkpoint, several milliseconds before contact, and continue
to the original 60 ms endpoint. This avoids comparing differently sagged initial
chains. The no-elasticity case is an artificial control, not a material model.
Rigid segments run for 60 ms from a 6 mm surface gap. Their effective contact mass
and global constraints differ from the chain: their trajectory is not ground
truth for the flexible thread.

Interpretation was set before running the matrix:

- A rigid failure shows elasticity is not necessary for that failure.
- Rigid passes and a no-elasticity-chain failure point toward articulated
  contact/conditioning, rather than requiring an elastic-energy explanation.
- Only DER failing would focus attention on elastic/contact interaction.
- Unit-dependent results mean the implementation is not numerically invariant
  under rescaling, even when the intended physical model is unchanged.
- None of these outcomes alone identifies a particular engine bug or proves
  that every MuJoCo contact model is unsuitable.

## Rescaling is an explicit part of this test

We must not assume that the earlier bending unit check also validates contact.
Every fixture runs at two timesteps, 156.25 ns and 78.125 ns, in three internal
unit systems. All measured trajectories are converted back to metres.

| Internal units | Length units/metre | Mass units/kg | Isolates |
| --- | ---: | ---: | --- |
| mm–g–s | 1,000 | 1,000 | Current baseline |
| mm–µg–s | 1,000 | 1,000,000,000 | Changing the mass-unit multiplier alone |
| 0.1 mm–g–s | 10,000 | 1,000 | Changing the length-unit multiplier alone |

Seconds, radians and physical material parameters stay fixed. Forces scale by
`M L`, inertias/torques/energies by `M L²`, and moduli by `M/L`; geometry, gravity,
contact widths and dimensional friction coefficients receive their own
conversions. See the complete [unit contract](RESCALING.md#unit-contract).
Numerical solver tolerance stays 1e−12; its unit sensitivity is being tested,
not presumed absent. Newton uses 100 iterations, an elliptic cone, `implicitfast`,
friction 0.3, 10 µs contact time constant and impedance 0.9999.

Before stepping, a compiled-model audit compares SI dimensions, inertia, mass,
gravity, contact parameters, initial shape and energy. Relative differences must
be below 1e−10. The 0.1 mm system preserves exactly representable segment spacing;
the earlier cm-system composite rounding is therefore not mixed into this test.
Passing this audit rules out the checked conversion discrepancies, not subsequent
solver conditioning or roundoff amplification.

The original gates remain: **less than 0.01 µm across unit systems**, **less than
1 µm across timesteps**, finite state, no warnings, full common duration and actual
contact. Shape differences use all recorded centerline points at matched 0.1 ms
timestamps. This sampling does not bound errors between snapshots. Contact force
and peak penetration are inspected at every physics step; the existing 2 µm peak
penetration limit is reported separately. No settled-contact claim is made.

Free-segment momentum is checked against accumulated support and gravity
impulses. Constraint work is integrated with average pre/post-step velocity and
compared with mechanical-energy change. This work residual includes quadrature
and integration error; it is not an exact energy-conservation proof. Root-clamp
reaction impulses are not reconstructed for the chain. The C++ batch logger calls
the same `mj_step1`/`mj_step2` sequence as the Python harness; it is not a new solver.

## Measured results

CPU MuJoCo 3.12.0 on Apple Silicon; corrected DER plugin unchanged from its earlier
benchmark. [Complete 24-case manifest](CONTACT_ISOLATION_RESULTS.json).
All figures below are numerical trajectory disagreements in µm.

| Fixture | Timestep difference in mm–g–s, 156.25→78.125 ns | Mass-unit difference at 78.125 ns | Length-unit difference at 78.125 ns |
| --- | ---: | ---: | ---: |
| Rigid normal drop | **2.234**, fail | 1.74e−10, pass | 0.000346, pass |
| Rigid angled/sliding drop | 0.345, pass | 2.06e−9, pass | 0.000269, pass |
| Chain without elasticity | **165.491**, fail | **10.546**, fail | **12.319**, fail |
| Flexible DER chain | **52.031**, fail | **3.502**, fail | **3.055**, fail |
| Required limit | <1 | <0.01 | <0.01 |

The comparisons in other unit systems are retained too. DER timestep disagreement
was 18.431 µm in mm–µg–s and 39.537 µm in 0.1 mm–g–s; neither passes. At the
coarser step, DER mass-unit and length-unit disagreements were 39.077 and 25.988 µm.
Selecting the most favorable unit system does not establish convergence.

The compiled-model unit audit passed, with maximum normalized difference
4.65e−16. No physical mass, radius, inertia floor, armature or damping was added.
Thus the checked physical-parameter conversion is consistent, while articulated
trajectories remain sensitive to numerical units. Possible contributions include
finite precision, solver thresholds and amplification of small perturbations;
this experiment does not separate them. It cannot compare the chain directly to
unscaled SI, which still hits the original minimum-inertia compilation limit.

All 24 cases completed without warnings or nonfinite state and reached contact.
Peak penetration stayed below 1.948 µm, passing the existing 2 µm peak gate;
settled penetration remains untested. Maximum Newton iterations were 18 of 100.
Free-body linear impulse residuals were below 1.15e−11 relative to accumulated
gravity/support impulses. Work residuals approximately halved with the timestep,
but these consistency observations do not override the failed trajectory gates.
Of 28 timestep/unit comparisons, 11 passed and 17 failed.

Both chains use the same fine-run pre-impact checkpoint. Therefore the 52.031 µm
DER result is a new controlled comparison, not a replacement for the earlier
14.379 µm comparison whose coarse run started at time zero. The restarted fine
trajectory matches the original fine trajectory at every saved position exactly
over 30–60 ms; [restart check](CONTACT_RESTART_CHECK.json). Batch and Python steps
also match bit for bit in rigid and DER contact regression tests.

The normal-drop follow-up did not establish convergence: 78.125→39.0625 ns changed
the trajectory by **4.502 µm**, and 39.0625→19.53125 ns by **1.135 µm**. Both fail
the unchanged 1 µm gate; the sequence is not monotonic. See the
[retained refinement manifest](CONTACT_NORMAL_REFINEMENT.json). Further blind
timestep refinement is stopped. This demonstrates the failed tested pairs,
not a proof that arbitrarily fine timesteps could never converge.

![Recorded timestep and unit comparisons](../../previews/neural_insertion/contact_isolation_v1/quality_review.png)

### What this changes

The contact problem does not require rod elasticity. A better bending/twisting
force law alone cannot address every observed failure. Rescaling is also part
of the dynamic sensitivity of the articulated models, despite correct parameter
conversion. A passing cantilever or visually plausible rebound is insufficient.

The subsequent [focused contact-solve audit](CONTACT_AUDIT.md) followed this plan: retain the
rigid counterexample, compare its impact against an independent integration of
the same contact law, and locate the first divergent contact impulse under time
and unit changes. For the chain, measure sensitivity to tiny matched physical
perturbations to separate roundoff amplification from a specific solver defect.
Patch only an identified defect with a regression test. Do not start a broad
MuJoCo fork or change the material model on this evidence alone. Discuss these
failed gates before expanding the simulation.

Matrix wall time was **351.913 s** for **1.08 s aggregate simulation**; the two
additional rigid runs took **7.435 s** for **0.12 s simulation**. Wall includes
construction, stepping and telemetry and overlapped repository tests; it is not
isolated compute time. No training ran. All **46 repository tests passed** and a
5 s native static-viewer smoke test completed. Static and result images were
inspected; no motion video is used as evidence for a passing gate.

## Reproduce

The portable pre-impact state is in `assets/neural_insertion/contact_preimpact.json`,
with original trajectory and plugin provenance. No ignored prior run is required.
The local batch logger requires a C++17 compiler, alongside the DER plugin build.

```sh
uv run --locked python -m sixlegs.neural_insertion.contact_diagnostics preview \
  --output previews/neural_insertion/contact_isolation_recheck
uv run --locked python -m sixlegs.neural_insertion.contact_diagnostics run \
  --output outputs/neural_insertion/contact_isolation/recheck_v1
uv run --locked python -m sixlegs.neural_insertion.contact_view \
  --kind segment_oblique --seconds 5
uv run --locked --extra wind python scripts/report_neural_contact.py \
  outputs/neural_insertion/contact_isolation/recheck_v1 \
  previews/neural_insertion/contact_isolation_recheck
```

Use a fresh output directory. Failed comparisons exit 1 and retain all completed
cases, source/scene/trajectory hashes, SI telemetry and engine-unit states. No
robot integration or RL is authorized by a passing diagnostic alone: spatial
refinement, physical grasp/release and material/contact calibration remain.

![Inspected static diagnostic fixtures](../../previews/neural_insertion/contact_isolation_v1/initial.png)

## Targeted follow-up protocol

After the rigid normal drop failed the first timestep pair at 2.234 µm, we chose
exactly two further halvings, to 39.0625 ns and 19.53125 ns, in mm–g–s. This asks
whether that simple contact case improves regularly with time resolution, rather
than assuming its first failure proves an irreducible contact error. This was a
follow-up selected after seeing the first result; no physics or gates change.

```sh
uv run --locked python scripts/refine_neural_contact.py \
  outputs/neural_insertion/contact_isolation/validation_v1 \
  outputs/neural_insertion/contact_isolation/normal_refinement_v1
```
