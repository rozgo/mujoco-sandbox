# Evaluate symmetry before another gait fine-tune

The user asked whether this should use a symmetry reward, explicitly requesting
evaluation rather than implementation. **No new symmetry training and no change
to the selected controller.** The approved visible-step checkpoint remains
`limb_visible_steps_selected_seed2.pt`, SHA-256
`db210589a90d0fc677269deb14fe4936e0b2732f99c40769b81f6de04f858794`.

## What symmetry should mean here

Encourage matching behavior in *mirrored situations*: an FL injury and its FR
counterpart, with all observations and actions reflected. Do not force the two
sides of one damaged body to take equal steps or move together. The controller
can still compensate asymmetrically and choose a phase-shifted gait.

This is an auxiliary **mirror-consistency loss**, rather than another reward
for a hand-chosen leg timing pattern. Task-level symmetry and mirror losses are
studied in [Mittal et al., ICRA 2024](https://arxiv.org/abs/2403.04359), and the
[official RSL-RL implementation](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py)
supports this method. These sources motivate the option; they do not establish
that it will improve this damaged-robot policy.

The repository already has a tested implementation in
`src/adaptive_locomotion/symmetry.py`: swap left/right joints, reflect axial and
polar vectors correctly, swap the missing-joint mask, reflect the range rays,
and compare reflected action means with a detached target. The loss masks
inactive joints. Recent visible-step and rear-overlap training used
**`symmetry_weight=0`**. Earlier bilateral training used weight 1, followed by
several later fine-tunes without the loss. This history alone does not prove
why the current mismatch emerged.

## Read-only physical probe

Run the frozen policy normally and through a diagnostic-only wrapper:

```text
pi_reflected(s) = mirror_action(pi(mirror_observation(s)))
```

The same unmodified weights drive the actual damaged body through its normal
torque-limited MuJoCo dynamics. The wrapper is not deployed or substituted in
the viewer. Seed **9161**, **0.5 ms physics / 20 ms control**, eight complete
trials per method on each of the four front-removal bodies. **All 64 tasks
complete with allowed support.** Timing below is measured from **only the first
trial** per body/method, using physical upward foot crossings at 1 cm clearance.

| Body | Original rear phase separation | Reflected-policy separation | Original simultaneous rear swing | Reflected simultaneous rear swing |
| --- | ---: | ---: | ---: | ---: |
| Lower FL | 41.6% | 10.8% | 0% | 6.91% |
| Lower FR | 10.4% | 41.9% | 6.91% | 0% |
| Whole FL | 39.0% | 6.5% | 0% | 7.82% |
| Whole FR | 6.4% | 41.0% | 8.00% | 0% |

Separation is 0% for simultaneous steps and 50% for halfway-apart steps.
Reflecting the existing FL behavior onto the FR body produces clearer rear
alternation. Reflecting the policy on FL transfers the worse FR timing to FL.
This supports a learned left/right discrepancy, rather than an unavoidable
physical limitation of the FR body. It does **not** prove a symmetry fine-tune
will learn the better of the two solutions.

On a separate bank of 1,080 fixed observations, mean per-case action mirror
RMSE is **0.385 normalized action units** (0.249 healthy, 0.452–0.510 across
front-removal cases). The fixed-state audit uses its existing 2 ms rollout
default; it is distinct from the 0.5 ms physical comparison above. Do not compare
this RMSE directly with historical audits collected on different states.

## Recommendation, not yet run

Evaluate a small mirror-loss fine-tune from the approved visible-step policy,
retaining the same lift and healthy-walk objectives. Mirror all corresponding
damage cases rather than special-casing FR. Compare against an equal-duration,
otherwise matched continuation with weight zero before attributing improvement
to symmetry. A first bounded pair of 90-second trials fits the fast-RL workflow.

Keep the good FL timing, all-foot lift, healthy gait and physical task checks as
gates. A consistency loss can also spread the worse gait to both sides; lower
mirror error alone is not success. Measure phase timing over all development
trials, then use an untouched final seed only after a candidate passes. No new
training has been started for this recommendation.

Reproduce from `experiments/adaptive_locomotion`:

```sh
uv tool run --from uv==0.12.12 uv run --locked python scripts/evaluate_symmetry_option.py
```

Raw evidence: [physical probe](SYMMETRY_OPTION_EVALUATION.json),
[fixed-state action audit](SYMMETRY_CURRENT_AUDIT.json), and the preceding
[overlap-only experiment](REAR_OVERLAP_TRAINING.md).
