# Contact smoothing: what fixes the disagreement, and what remains

**Two numerical choices in the baseline fixture caused most of the failed
contact gates: a contact force that switches on as a step at entry, and a
first-order integrator through a 5–10 µs impact. Replacing them with a
penetration-ramped impedance and MuJoCo's RK4 integrator, with no other change,
passes every rigid gate by 10–100× and the low-velocity settling gates by
orders of magnitude. The 6 mm frictional chain drop still fails the 1 µm
pointwise gate after its first impact, because that trajectory amplifies
roundoff-level differences; a third timestep halving and tiny perturbations
show this is sensitivity of the scenario, not solver error. No engine patch is
involved and no physical calibration has been established.**

This follows the [contact audit](CONTACT_AUDIT.md). Fixtures, pre-impact
checkpoint, masses, dimensions, inertia, friction, solver tolerance and unit
systems are unchanged; the variant audit confirms that only `solimp`,
`solref` and the integrator differ from the baseline models. CPU MuJoCo 3.12.0.

## Mechanism

The pinned engine computes, per contact normal, `a_ref = −B·v − K·d(r)·pos`
with `B = 2/(d_width·τ)`, `K = 1/(d_width·τ)²` and regularizer `R = (1−d)/d·diagA`.
The baseline uses `solimp = (0.9999, 0.9999, …)`, so `d` is constant and the
damping term `B·v` acts in full at the first penetrating step. A grazing
contact at picometre depth therefore applies a finite impulse, and the audit's
pair probe showed exactly such an event starting the unit divergence. With
`solimp = (0.0001, 0.9999, width, 0.5, 1)` the impedance ramps linearly from
MuJoCo's minimum over `width`, so the force is continuous in state at entry
(down to the 1e−4 impedance floor). This is the documented impedance ramp,
not a new law; it remains an uncalibrated numerical regularization.

Separately, `implicitfast` is first order in time. The independent SI
recurrence shows its error through the impact is O(Δt) and about 2 µm at
78 ns, even when the entry time is localized exactly. MuJoCo's RK4 evaluates
collision and the constraint solve at four stage states, so with a continuous
law its error through the impact falls below 0.1 µm at 156 ns.

| Law | `solimp` | `solref` | Rigid peak penetration (normal / angled) | Rebound after a 6 mm drop |
| --- | --- | --- | ---: | ---: |
| `flat` (baseline) | 0.9999 0.9999 1 µm 0.5 2 | 10 µs, ratio 1 | 1.25 / 1.95 µm | 109 µm |
| `ramp_5us` (candidate) | 0.0001 0.9999 1 µm 0.5 1 | 5 µs, ratio 1 | 0.97 / 1.23 µm | 192 µm |

The ramp is softer at shallow depth, so the 10 µs variant exceeded the 2 µm
penetration gate on the angled drop (2.19 µm) and was not selected. The
shorter time constant restores penetration below 1.3 µm and keeps `refsafe`
satisfied (τ ≥ 2Δt). The candidate rebounds higher; both rebounds are
numerical properties of uncalibrated laws, not thread measurements.

## Rigid fixtures: gates pass with margin

Timestep disagreement, 156.25 → 78.125 ns, µm (gate < 1); identical in all
three unit systems. Unit disagreement was below 1e−6 µm for every candidate case.

| Fixture | flat + implicitfast | flat + RK4 | ramp + implicitfast | **ramp + RK4** |
| --- | ---: | ---: | ---: | ---: |
| Normal drop | 2.234 | 1.134 | 5.867 | **0.077** |
| Angled, sliding drop | 0.345 | 0.210 | 1.293 | **0.025** |

Against the independent continuous reference for each law, the candidate's
normal-drop error is 0.08 µm at 156 ns and 0.005 µm at 78 ns; the independent
RK4 recurrence reproduces MuJoCo to 1e−4 µm, as the Euler recurrence did for
the baseline. Same-state MuJoCo forces under the ramped law match the
independent formula to 1.6e−11 relative across units.

## Flexible chain: one gate fixed, one gate reinterpreted

Same 30 ms after the shared pre-impact checkpoint; tip impact at 0.34 m/s.

| DER chain comparison | Baseline | Candidate (ramp + RK4) |
| --- | ---: | ---: |
| Timestep 156.25 → 78.125 ns | 52.03 µm | 8.03 µm |
| Timestep 78.125 → 39.0625 ns | not run | 6.52 µm |
| Mass units, fine step | 3.50 µm | **0.0031 µm** |
| Length units, fine step | 3.06 µm | **0.0099 µm** |
| Length units, coarse step | 25.99 µm | 0.051 µm |
| 1e−12 rad perturbations, seeds 7/17/29 | 3.6 / 16.5 / 11.8 µm | 0.55 / 0.55 / 0.55 µm |
| Peak penetration | 1.74 µm | 1.09 µm |

Controls isolate the ingredients: the ramp with `implicitfast` gives 54 µm and
the flat law with RK4 gives 5.8 µm; the integrator matters more for the chain
and the ramp matters more for unit invariance. Without friction the candidate's
timestep disagreement is 1.6 µm, so stick–slip switching is the largest
remaining amplifier.

The time profiles explain the residual. Immediately after the first impact the
candidate's coarse and fine trajectories differ by 0.0045 µm, 10–1000× less
than the baseline. The difference then grows roughly exponentially, with an
e-folding time near 3 ms, and jumps at discrete grazing events. Unit pairs and
the seeds start at 1e−11 µm and grow the same way. A third halving improves the
early error about 7× but leaves the 30 ms maximum unchanged. A pointwise 1 µm
agreement over 30 ms of a frictional multi-impact slap therefore demands
sub-nanometre accuracy at impact from any solver; it measures the scenario's
sensitivity rather than numerical error. The chain without elastic forces
remains far more sensitive (12–25 µm) and is retained as a control.

## Low-velocity settling fixture

The surgical task places thread at millimetres per second; it does not drop it
6 mm. A new `der_settle` fixture releases the same straight DER rod at rest
with the support 50 µm below its lower surface, so the tip lands near
0.03 m/s and the rod lies down and settles over 60 ms. With the candidate:

| Settling comparison | Result | Gate |
| --- | ---: | --- |
| Mass-unit and length-unit disagreement, coarse step | about 1e−11 µm | < 0.01 µm, pass |
| Timestep 156.25 → 78.125 ns | 6.6e−5 µm in the development run; recorded value follows | < 1 µm, pass |
| Peak penetration | 0.227 µm | < 2 µm, pass |
| Settled penetration after 60 ms | 0.04–0.05 µm, no rebound | < 0.2 µm, pass |

This is the first fixture in which the settled-penetration gate could be
measured at all. Maximum centerline speed decays from 9.5 mm/s at 10 ms to
1.9 mm/s at 60 ms; the rod is settling, not yet at rest.

## Decision

Adopt the candidate law and integrator for the thread fixtures; they are
ordinary MuJoCo settings, documented here as a changed numerical model. Keep
all gates unchanged, and apply the pointwise 1 µm / 0.01 µm gates where a
converged trajectory is physically meaningful: the pre-impact window, the first
impact and low-velocity contact. For impact-rich scenarios, report impact-time,
impulse, peak-force and rest-state convergence in addition to the pointwise
measure, and state the sensitivity measured with predeclared perturbations.
Measured contact and material parameters are still required before any
accuracy claim about a real thread. RK4 costs about four forward evaluations
per step; at equal accuracy it is cheaper than halving the implicit step. The
thread still needs 100 ns steps, so coupling to the 1 ms robot clock remains
an open design question before robot integration. Robot thread handling and RL
remain gated.

## Reproduce

```sh
uv run --locked --extra wind python -m sixlegs.neural_insertion.contact_smoothing run \
  --output outputs/neural_insertion/contact_smoothing/recheck_v1 --jobs 8
uv run --locked --extra wind python scripts/report_neural_contact_smoothing.py \
  outputs/neural_insertion/contact_smoothing/recheck_v1 \
  previews/neural_insertion/contact_smoothing_recheck_v1 \
  --isolation outputs/neural_insertion/contact_isolation/validation_v1 \
  --sensitivity outputs/neural_insertion/contact_audit/sensitivity_v1
uv run --locked --extra wind pytest -q tests/test_neural_contact_smoothing.py
```

Use a fresh output directory; the run exits 1 if any gated candidate comparison
fails, which the 6 mm chain drop currently does. Numbers above come from the
development runs that preceded the recorded matrix; the recorded manifest
(`CONTACT_SMOOTHING_RESULTS.json`) and inspected plots
(`previews/neural_insertion/contact_smoothing_v1/`) are added by the follow-up
checkpoint once that run completes, with any differences noted there.
