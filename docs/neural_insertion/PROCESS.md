# How we got here: process and corrections

A running account of the thread-insertion project: what we tried, what each
result told us, and where our own approach had to be corrected. Measurements
live in the linked reports; this page explains the reasoning between them.
Updated at each milestone.

## 1. Static workcell (October 5)

We built the five-slide gantry, microscope, phantom, vessels, targets and thread
cassette as a static scene first, and checked clearance and limits before any
motion. [README](README.md).

## 2. The thread would not compile, then would not converge

A 40 µm, 44 mm thread at SI scale hit MuJoCo's minimum-inertia limit. Consistent
unit rescaling (mm–g–s) fixed compilation without changing the physics, and
bending matched the analytical cantilever within 2.4%. But a 6 mm drop onto a
support changed by 35 µm when the timestep was halved. [Rescaling](RESCALING.md).

## 3. A better rod model did not fix contact

We ported the published discrete elastic rod (DER) model. The port exposed a
real defect in the published force mapping (zero restoring torque for pure
twist), which we corrected and verified against the energy gradient. Contact
still failed refinement. [DER validation](DER_VALIDATION.md).

## 4. Narrowing the cause by removing ingredients

A single rigid segment also failed, and a chain with no elastic forces failed
worse, so elasticity was not required for the failure. Changing only the mass
or length unit also changed chain trajectories, although compiled physics
agreed. [Isolation](CONTACT_ISOLATION.md).

## 5. Checking the engine against an independent calculation

Same-state contact forces matched an independent formula to 1e−11 and agreed
across units, so the force computation was not the problem. An independent
fixed-step recurrence reproduced MuJoCo's rigid rebound error exactly, which
moved suspicion from the engine to time integration. In the chain, the first
divergence was two trajectories landing picometres on opposite sides of contact
activation. [Audit](CONTACT_AUDIT.md).

## 6. Two causes, two standard fixes

The independent model separated two effects. The baseline contact law switches
on a velocity-proportional force as a step at zero penetration, so a grazing
entry gets a finite impulse; MuJoCo's impedance ramp removes the step. The
default integrator is first order through a 5–10 µs impact; RK4 removes that
error. With both, rigid gates passed by 10–100× and chain unit agreement improved
about 1000×. [Smoothing](CONTACT_SMOOTHING.md).

## 7. Correction: a stress test had become an acceptance gate

The 6 mm chain drop still failed its 1 µm gate. Its error grew exponentially
after impact regardless of timestep, and 1e−12 rad perturbations grew the same
way, so it measured the scenario's sensitivity rather than solver error.

When asked whether that gate was real, we traced it back. It had been chosen in
the rescaling study as a convenient numerical stress test and was then carried
through four studies as if it were a requirement. No task phase drops the
thread 6 mm and needs a 1 µm repeatable landing; every precision event is a
slow, millimetre-per-second contact. A low-velocity settling fixture built for
the task regime passed every gate by three to nine orders of magnitude.

What we changed in our process as a result:

- Derive gates from task phases first; keep stress tests, but label them.
- When a gate fails after refinement, test whether it measures error or
  sensitivity (perturbation seeds, refinement trend) before more tuning.
- Change one numerical choice at a time and keep the controls, so a fix can
  be attributed rather than inferred.
- Keep an independent reference that never calls the engine under test.

Smaller execution mistakes we corrected along the way: a scratch naming bug
that overwrote early result files (rerun before any result was used), a batch
launcher that exceeded the shell's line limit, and a test that wrongly assumed
the impedance ramp gives exactly zero entry force (MuJoCo's impedance floor is
1e−4). [Task-derived acceptance](ACCEPTANCE.md).

## 8. The task regime is far cheaper than the stress test (October 6)

The 5 µs contact time constant, and its 100 ns step, had been chosen to keep
penetration small in a 0.34 m/s drop. At task speeds the same penetration gates
hold with a time constant four times longer, so we measured the largest step
that still passes: 10–20 µs on the settling fixture. [Task regime](TASK_REGIME.md).

We then built sliding, loading and release fixtures. The first sliding design,
a vertical needle pushing the resting thread sideways, jammed: a 150 µm needle
tip wedges a 40 µm thread against the support. We replaced it with a preloaded
needle sliding across the thread, which is how a withdrawing needle actually
touches it; the thread then rolls half the needle travel, as it should.

Release failed the timestep gate at 10 µs (1.58 µm) because its 150 µm fall
lands faster than settling does. Refining to 5 µs passed (0.19 µm). We kept the
failed comparison and adopted 5 µs for every thread phase, 64× larger than the
step the drop needed. The press fixture also showed that contact compliance
under load is a solver property, about 1 µm at 100 µN, which calibration must
address before forces are trusted.

## 9. The robot's first motion (October 6)

With gates fixed beforehand, we gave the gantry a programmed servo and asked
it to hold, then hover 1 mm above each target. The first inverse kinematics
failed immediately: Z travels only 8 mm down, so the design reaches the surface
with the needle slide. The first approach then showed a 265 µm tracking peak,
which turned out to be the needle hanging 0.2 mm past its retracted stop at
t = 0. Carrying both slides 0.5 mm out fixed it. Every gate then passed with
large margins. [First motion](MOTION.md).
