# MuJoCo DER acceptance experiment

The first candidate is the published `adapteddlo_muj` rod on MuJoCo 3.12.0 CPU.
The approved workcell and native-cable results remain unchanged. Source revision,
license and file hashes are in `src/sixlegs/neural_insertion/der_vendor/`.

## Minimum evidence, fixed before the validation run

- Same mass, radius, inertia, contact settings and physical units as the baseline;
  no extra mass, armature, supports or damping to mask instability.
- Repeated evaluation, copy, state restoration and reset must reproduce forces.
- Bending/twist forces must agree with the gradient of the declared elastic
  energy within **0.01%** relative norm, checked at two finite-difference steps.
- Retain **3%** preliminary cantilever error, **1 µm** matched-time trajectory
  convergence, **2 µm** peak penetration and **0.2 µm** genuinely settled
  penetration gates. A rebound window does not establish settled contact.
- In a no-load, no-gravity, undamped relaxation, mechanical energy drift should
  decrease with timestep refinement and remain below **0.1%** on the tested window.
- Passing these numerical fixtures permits a physical tool/thread hold-release
  test and bounded approach. It does not establish tissue insertion, calibrated
  material response or real-world micrometre accuracy. RL follows basic plant
  validation, not a simulator compilation alone.

## Port and variants

The source plugin targets 3.3.2. The port builds a separate dynamic library using
the installed 3.12 headers and library. Explicit bodies replace its custom XML
composite; the engine itself is unchanged. A massless fixed endpoint frame adds
no DOF, contact or support. Eigen is a pinned, hash-checked local build dependency.

Compatibility/lifecycle changes initialize curvature and twist, compute current
curvature before transporting frames, use a signed `atan2` twist angle, and put
twist-unwrapping history in MuJoCo plugin state with copy/reset/advance support.
Extra `mj_forward` calls do not advance that history.

Two named force variants are retained:

- `published`: upstream Cartesian elastic forces and lever-based joint torques.
- `direct`: the same Cartesian elastic forces projected with MuJoCo's Jacobians,
  plus the material-frame endpoint moments from the torsional energy. This is a
  local correction, not an unchanged reproduction of the published implementation.

The audit found zero restoring torque for a straight twisted rod in
`published`, despite positive twist energy. Adding only an endpoint torque did
not fix combined bending/twist virtual work. Direct nodal force projection plus
endpoint moments passes the recorded energy-gradient checks. This diagnoses the
retained force mapping in our compatible port, not every DER model or all results
in the paper. The original upstream source files remain byte-for-byte preserved.

Both variants use an inextensible, circular, initially straight rod and
quasistatic twist. The code's effective twist length excludes the first/last
half-edge contributions; record this length and refine it with segment count.
Quasistatic twist suitability for fast contact/release is still a physical model
question. The benchmark material parameters remain illustrative.

[Published model](https://arxiv.org/html/2310.00911v3),
[pinned implementation](https://github.com/qj25/adapteddlo_muj/tree/71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad).

## Recorded results

The complete first physics suite remains **not accepted for robot integration or
RL**. [Full manifest](DER_RESULTS.json), including source/build hashes, all gates,
the failed upstream-force audit, and per-case configuration/timing.

| Check | Corrected `direct` result | Gate |
| --- | --- | --- |
| Force versus elastic-energy gradient | Maximum relative error 2.85e−9 across six poses and two difference steps | Pass, <1e−4 |
| Repeated evaluation, copy, state restore, reset | Zero force difference in the recorded fixture | Pass |
| Unit-scaled force equivalence | Relative difference 1.01e−15 | Pass, <1e−9 |
| Native mass/inertia preservation | Identical compiled values | Pass |
| 64-segment cantilever | 258.892 µm; 2.400% error against 265.258 µm analytical reference | Preliminary pass, <3% |
| 32→64 cantilever refinement | 6.110 µm; 2.304% of reference | Preliminary pass, <3% |
| Undamped free relaxation, 2→1 µs step | 0.0998 µm maximum shape difference over 10 ms | Pass, <1 µm |
| Free-relaxation energy drift | 0.0749%→0.0380%, decreasing with step | Pass, <0.1% |
| Frictional contact, 312.5→156.25 ns | 58.573 µm maximum shape difference over 60 ms | Fail |
| Frictional contact, 156.25→78.125 ns | 14.379 µm maximum shape/tip difference | Fail, <1 µm |
| Additional contact refinement, 78.125→39.0625 ns | 41.762 µm maximum shape/tip difference | Fail; improvement did not persist |
| Peak support penetration across three contact runs | 1.744 µm | Pass, <2 µm |

The native-cable baseline differed by 34.979 µm at the same final timestep pair.
Here "same pair" means 156.25→78.125 ns. The apparent improvement at that pair
did not survive further refinement. These are disagreements between numerical
trajectories, not measured errors against a real thread. All completed cases were finite with
zero MuJoCo warnings. Contact solver iterations peaked at 18 including the extra
refinement, below the 100 cap.

The additional run is retained in [its manifest](DER_CONTACT_REFINEMENT.json).
It halved only the timestep, preserving the plugin, units, initial state, material
and contact law, and took 179.68 s wall time for 60 ms simulation. Further timestep
reduction is stopped at this failed convergence check.

![Measured DER checks including the failed finer contact step](../../previews/neural_insertion/der_v2/quality_review.png)

Outstanding: contact spatial/unit refinement, actually settled penetration,
physical grasp/release, the full published buckling fixtures, and material/contact
calibration. The free-relaxation test releases an initially bent rod from rest;
it is not a validated gripper release. These results justify further contact
diagnosis, not tissue insertion or learned thread handling.

The follow-up [contact-isolation study](CONTACT_ISOLATION.md) implements rigid
segments, a chain without elastic forces, and the matched DER chain. It explicitly
checks mass-unit and length-unit changes, impulse balance and constraint work.
The rigid segment is a diagnostic control; our intended thread remains flexible.
These checks narrow the cause but do not alone identify an engine defect. A broad
engine patch or another rod-energy change is not justified by the current evidence.
Discuss the failed gates before expanding to tool contact or another solver.

## Reproduce and inspect

Requires a C++17 compiler; the first build downloads only the pinned Eigen source
archive into ignored build storage. No engine fork, global install or Python
dependency change is required. CPU Apple Silicon was tested. Linux build support
is present but has not been exercised.

```sh
uv run --locked python -m sixlegs.neural_insertion.der_build
uv run --locked python -m sixlegs.neural_insertion.der_view --static
uv run --locked python -m sixlegs.neural_insertion.der_benchmarks run \
  --output outputs/neural_insertion/der/recheck_v1
uv run --locked --extra wind python scripts/report_neural_der.py \
  outputs/neural_insertion/der/recheck_v1 previews/neural_insertion/der_recheck_v1
```

The saved additional refinement can be reproduced with:

```sh
uv run --locked python scripts/refine_neural_der.py \
  outputs/neural_insertion/der/validation_v2 outputs/neural_insertion/der/refinement_recheck_v1
```

The benchmark exits **1** for failed quality gates. Use a fresh output directory.
Native `--seconds 5` inspection and all 42 repository tests passed. The initial
validation attempt hit a NumPy-boolean JSON reporting error after bending; its
manifest and trajectories are retained under `validation_v1`. The fixed complete
run is `validation_v2`. No numerical failure was erased or reclassified as success.
