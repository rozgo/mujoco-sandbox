# Foot and terrain friction review

The user suggested that higher foot friction could improve the natural-looking
terrain behavior, then explicitly pointed out that terrain/support friction
should also be considered. We tested both together while keeping the policy
frozen. The accepted results video retains its original physics and footage.

The current feet and terrain both use sliding friction **0.8**, equal geom
priority, three contact dimensions, pyramidal friction cones, `impratio=1` and
`noslip_iterations=0`. There are no explicit contact pairs overriding these
values. MuJoCo takes the component-wise maximum friction when the two geoms have
equal priority, so increasing either side would increase this contact's sliding
coefficient. We set both sides consistently for the comparison.
[Official contact combination rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).

Recorded-state kinematics confirm contact-point slip: **0.25–1.25 cm/s mean**
across the eight terrain conditions during the 2–10 s hold. This measures
tangential velocity at active foot-to-terrain contacts with recorded terminal
load above 5 N, sampled at 50 Hz; it is not inferred solely from body drift or
foot-center motion. See [the slip measurements and method](TERRAIN_SLIP_REVIEW.json).

## Matched comparison

We set **both allowed terminal sliding friction and physical static support
sliding friction** to 0.8, 1.0 or 1.2 before constructing the batched physics
models. Other link coefficients, torsional/rolling friction, solver parameters,
geometry, joint limits, torques, commands, reset perturbations, weights and
acceptance gates stayed fixed. This is an optional environment parameter;
the default remains unchanged.

Each setting ran all eight reviewed healthy terrain conditions and three passing
controls (flat walk–stop–walk, high pads, front-right gap), with four matched reset
trials each. All **44 baseline result rows** exactly reproduce the original CPU
evaluation. Seed 9307 is reused for diagnosis; this is not a new independent
holdout or evidence of additional learning.

| Foot + support friction | Strict terrain passes | Mean peak hold drift | Unintended-support trials | Passing controls |
| --- | --- | --- | --- | --- |
| 0.8 | 0/32 | 12.33 cm | 17/32 | 12/12 |
| 1.0 | 0/32 | 12.43 cm | 16/32 | 12/12 |
| 1.2 | 1/32 | 12.09 cm | 15/32 | 12/12 |

All **132 trials remain upright** and respect torque caps. Maximum sampled
penetration is below 8 mm for all three settings. The one additional strict pass
at 1.2 is on extreme pads. Their mean peak drift improves from 17.46 to 15.44 cm,
but some other conditions worsen: rear-right-gap drift rises from 7.87 to 9.73 cm.
Across the eight terrain groups, mean peak drift falls only about **2%** at 1.2.

Increasing friction alone within this range does not resolve most flags. The
foot finding an adjacent pad, body tilt and non-foot support also contribute.
MuJoCo's contact regularization can produce slow creep even when a Coulomb
friction limit is adequate; the coefficient is only one contact-model parameter.
That is a possible further explanation here, not established causation from
this sweep. A future comparison could investigate contact regularization with
the same fixed-policy and matched-trial method.
[Official slip diagnosis](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip).

No new friction preset is promoted based on this small improvement. All
counterfactual traces, models and result rows are retained, and the user-accepted
original terrain motion is included in [results video v3](RESULTS_VIDEO_V3.md).

## Reproduce and validation

From the repository root:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked \
  python experiments/adaptive_locomotion/scripts/compare_standing_friction.py
```

[Raw comparison](TERRAIN_FRICTION_SWEEP.json) includes source commit, checkpoint
hash, per-trial metrics, model/trajectory hashes and exact settings. The CPU
MuJoCo/mjbatch comparison and capture took **23.742 seconds**, representing
**1,344 aggregate simulated world-seconds**. **No training was performed.**

A baseline assertion caught an initial script wiring error: the transition flag
was omitted for the flat control, so it held still rather than walking, stopping
and resuming. The first eight baseline groups reproduced correctly before that
check stopped the run. The omitted argument was fixed in `8c12c70` and the entire
matched sweep was rerun. The discarded setup attempt consumed 368 additional
aggregate simulated world-seconds; its wall time was not separately recorded.
It is excluded from the comparison table and the 23.742 s final sweep timing.

Focused standing checks: **22 passed, 5 CUDA-only skipped, one existing training
smoke deselected**. The new check verifies that both feet and terrain reach 1.2
inside the batch's model storage and actual contact records, while initial
states, actuator caps and other friction dimensions stay identical. Default
behavior is additionally checked by the exact 44-row baseline reproduction.
The friction sweep used CPU physics; it does not claim a measured Warp result.
