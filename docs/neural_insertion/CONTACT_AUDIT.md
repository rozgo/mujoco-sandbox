# Contact audit: what causes the disagreement?

**The tested contact forces are consistent. Fixed-step impact timing and strong
trajectory sensitivity explain important parts of the failures. No specific
MuJoCo engine defect has been established, and the accuracy gates remain failed.**

This follows the [isolation tests](CONTACT_ISOLATION.md). We kept the flexible
DER model, physical mass, dimensions and contact law unchanged. The rigid segment
remains a control for understanding contact without elastic forces. Everything
ran on CPU MuJoCo 3.12.0; the installed engine was not patched.

## Why these checks

The isolation tests showed different trajectories, but did not reveal where the
difference began. This audit separates three questions:

1. Given the **same state**, do equivalent units produce the same force?
2. Does the rigid drop agree with an **independent calculation of the same law**?
3. Can **tiny state differences** change when contact starts and then grow?

We used 12 rigid contact states, seven saved states for each chain, an independent
normal-impact reference, two unit-paired trajectories inspected at every physics
step, and three predetermined tiny planar perturbations (seeds 7, 17, 29).
A separate zero-tolerance run tests sensitivity to solver termination. These
are diagnostic controls, not new accepted material or solver presets.

## Same-state forces agree

Across mm–g–s, mm–µg–s and 0.1 mm–g–s:

| Frozen-state comparison | Maximum relative difference/error |
| --- | ---: |
| Rigid normal acceleration versus independent formula | 1.16e−11 |
| No-elasticity chain contact generalized forces, across units | 7.30e−11 |
| Flexible DER contact generalized forces, across units | 9.49e−10 |
| Flexible DER accelerations, across units | 1.94e−10 |

Contact sets match in all sampled same-state comparisons. The chain also has
small differences in unconstrained acceleration, before contact forces are added.
This supports floating-point/conditioning differences as a source of small
perturbations; it does not isolate a particular floating-point operation.
The evidence does not support a large unit-conversion error in these force
calculations. It also does not cancel the failed trajectory unit-agreement gate.

[Force and independent-reference results](CONTACT_AUDIT_RESULTS.json).

## The independent rigid drop explains the timestep failure

For the horizontal capsule, the two endpoint contacts are symmetric, so the
normal contact solve has a closed-form solution. An independent SI implementation
was run in two ways: the same semi-implicit Euler recurrence as this rigid fixture,
and adaptive DOP853 integration of the continuous-time contact law after exact
ballistic flight. It does not call MuJoCo for forces or integration.

| MuJoCo timestep | Difference from continuous reference, µm | Difference from independent fixed-step recurrence, µm |
| --- | ---: | ---: |
| 156.25 ns | 6.779 | 0.000170 |
| 78.125 ns | 4.594 | 0.000152 |
| 39.0625 ns | 0.0929 | 0.0000349 |
| 19.53125 ns | 1.228 | 0.000113 |

The independent fixed-step calculation reproduces the troublesome rebounds.
The continuous reference changes by only **2.65e−8 µm** when its step bound and
tolerances are tightened. These comparisons use the same saved 0.1 ms timestamps;
they do not bound errors between snapshots or establish physical material accuracy.

The first exact ballistic contact is at 34.974870839 ms. The four fixed-step runs
first apply contact about −27.089, −27.089, +11.973 and −7.559 ns from that time.
Semi-implicit free flight and the position of the impact within a timestep alter
the first impulse. The contact law turns on a damping force at contact; a small
entry-time change can therefore alter rebound. This accounts for the rigid case
without blaming an incorrect force calculation. The favorable 39 ns result is
not sufficient evidence of convergence: the next halving is worse.

### Independent formula and its limits

Let `s` be signed surface gap, `v` normal velocity, `d=0.9999`, `tau=10 µs`,
and `g=9.81 m/s²`. For this symmetric two-contact fixture:

```text
B = 2 / (d tau)
K = 1 / (d tau)²
a_ref = -B v -K d s
alpha = 2d / (1+d)
a = -g                                      when s > 0
a = -g + alpha max(0, a_ref + g)             when s <= 0
```

Each normal row has approximate inverse mass `1/m` and regularizer
`R=(1-d)/(d m)`. Equal endpoint forces satisfy `(2/m+R) f = a_ref+g` when positive;
the two forces give the expression above. Tangential forces and rotation vanish
by symmetry. The fixed-step reference uses `v_next=v+h*a`, followed by
`s_next=s+h*v_next`. This reduction does **not** apply to the bent, frictional
chain. Its force law is a numerical model, not a calibrated thread/tissue law.

Sources: the pinned [constraint construction](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_constraint.c),
[integration code](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_forward.c),
and [solver-parameter documentation](https://mujoco.readthedocs.io/en/3.12.0/modeling.html#solver-parameters).
[Source/library hashes](CONTACT_SOURCE_AUDIT.json) retain what was inspected.

## The chain amplifies small changes at contact entry

We advanced equivalent DER models together for the first 10 ms after the saved
30 ms pre-impact checkpoint. Every physics step was inspected, with denser 10 µs
telemetry saved. The first contact-set mismatch occurred at:

| Unit change | Time after checkpoint | Maximum shape difference just before that step |
| --- | ---: | ---: |
| Mass units | 8.544531 ms | 0.0000159 µm |
| Length units | 8.509609 ms | 0.0000533 µm |

In the mass-unit pair, the relevant capsule surfaces were **5.687 pm above** the
plane in one trajectory and **2.561 pm below** it in the other. The first run had
no contacts and the second had two. Re-evaluating either saved state in all three
unit systems gives the same contact set and closely matching forces. Thus this
event is caused by different evolved states crossing the activation boundary,
not by units alone making identical geometry collide differently.

The difference crosses the 0.01 µm unit gate shortly afterward, at 8.699 ms for
mass units and 8.540 ms for length units. Small differences already exist before
this event; this is the first differing contact set, not the first unequal bit.
Contact onset is a demonstrated amplification mechanism, not a complete proof
that every later difference has the same cause.

[Step-by-step results](CONTACT_PAIR_RESULTS.json),
[frozen event replay](CONTACT_EVENT_REPLAY.json).

The same-unit perturbations independently show sensitivity. We changed only the
initial planar joint rotations by a vector of norm 1e−12 rad, then stepped normally:

| Seed | Initial maximum shape change, µm | Maximum difference over 30 ms, µm |
| --- | ---: | ---: |
| 7 | 4.02e−8 | 3.623 |
| 17 | 2.17e−8 | 16.466 |
| 29 | 1.94e−8 | 11.761 |

These deliberately tiny numerical perturbations are not a model of real surgical
uncertainty and do not establish physical chaos. They show that the observed
trajectory sensitivity is not exclusive to changing units.

Setting solver tolerance to zero, with no state perturbation, changed the
trajectory by 2.332 µm, took 186.87 s wall instead of about 46 s, and reached the
100-iteration cap. This is not a converged reference. We did not run a new full
unit/timestep matrix at that setting, so it cannot be claimed to cure or fail all
those gates. [Sensitivity results](CONTACT_SENSITIVITY_RESULTS.json).

![Inspected contact-audit plots](../../previews/neural_insertion/contact_audit_v1/quality_review.png)

## Decision

Follow-up: the [contact-smoothing experiment](CONTACT_SMOOTHING.md) tested the
entry-localization idea in the independent reference and found a cheaper route:
a penetration-ramped impedance removes the entry step, and RK4 removes the
first-order impact error. Rigid gates pass with margin; the chain drop remains
sensitivity-limited. The text below is the decision as written before that work.

The audit narrows the next experiment: test **contact-event localization and
adaptive substepping** against the independent rigid reference while keeping the
same physical contact law, then repeat the flexible-chain gates if it succeeds.
That is a specific integration experiment, not a broad engine rewrite. No such
integrator was implemented or accepted in this audit. More digits, a different
mass unit or smaller fixed timesteps alone have not supplied the required guarantees.

We still lack a converged chain contact/release model and physical calibration.
The 1 µm refinement and 0.01 µm unit gates are unchanged. Robot thread handling
and RL remain gated; no engine patch is justified as a demonstrated bug fix yet.

## Reproduce

These commands consume the hash-checked saved isolation runs described in
[CONTACT_ISOLATION.md](CONTACT_ISOLATION.md). Use fresh output paths. The optional
wind extra supplies SciPy and matplotlib; no new dependency or global install is
needed. Audit commands report diagnostic completion; `ready_for_robot_or_rl`
remains false and is not a passing physics-acceptance result.

```sh
uv run --locked --extra wind python -m sixlegs.neural_insertion.contact_audit \
  outputs/neural_insertion/contact_isolation/validation_v1 \
  outputs/neural_insertion/contact_audit/recheck_v1 \
  --normal-refinement outputs/neural_insertion/contact_isolation/normal_refinement_v1
uv run --locked python -m sixlegs.neural_insertion.contact_pair_probe \
  outputs/neural_insertion/contact_audit/pair_recheck_v1
uv run --locked python -m sixlegs.neural_insertion.contact_sensitivity \
  outputs/neural_insertion/contact_isolation/validation_v1 \
  outputs/neural_insertion/contact_audit/sensitivity_recheck_v1
uv run --locked --extra wind python scripts/report_neural_contact_audit.py \
  outputs/neural_insertion/contact_audit/recheck_v1 \
  outputs/neural_insertion/contact_audit/pair_recheck_v1 \
  outputs/neural_insertion/contact_audit/sensitivity_recheck_v1 \
  outputs/neural_insertion/contact_isolation/validation_v1 \
  outputs/neural_insertion/contact_isolation/normal_refinement_v1 \
  previews/neural_insertion/contact_audit_recheck_v1
```

The accepted static robot and diagnostic viewers are unchanged. Frozen fixture:
`uv run --locked python -m sixlegs.neural_insertion.contact_view --kind der`.
Raw trajectories, partial development results and generated files remain in ignored
outputs; selected manifests and plots are retained here. No new motion video or
training was produced for this failed physics gate.
