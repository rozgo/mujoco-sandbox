# Motor learning: stand, walk and hover

Current scope: one MaleCNS actor, one physical fly, three motor commands. Utility
selection and needs-driven behavior are deferred. Takeoff, landing and transitions
between ground and air remain later requirements; the first pilot starts each
primitive from its declared ground or airborne state.

Current development choice for the next motor curriculum is
[ground_outcome_03](runs/ground_outcome_03/SUMMARY.md), which improves resting
wings in both ground commands. Its unchanged continuation04 was not better.
The same03 checkpoint drives all commands; there is no per-command policy
selection. Previous motor_focus_06 remains preserved. Full posture/tracking
gates and hover are still incomplete; this is not a released motor solution.

Every world uses `wing_motion`: 5 kHz native MuJoCo physics and 500 Hz control.
The checkpoint records a fingerprint of masses, inertias, joints, contacts,
actuation, solver settings and flight-force parameters. Native and batched
evaluation must match. Wing bodies have zero mass/spatial inertia, no collisions
and no aerodynamic terms. Independent diagonal angular armature permits wing
motion; the tested custom flight law is the only wing-to-body coupling.

## Declared first pilot

- Parent: preserved `motor_flight_angle_probe_01.pt` walking diagnostic.
- 397 inputs, 78 outputs, four internal graph updates, the same fixed measured
  connections and trainable cell dynamics. Two neutral altitude inputs extend
  the parent's 395-input encoder. No raw-sensor-to-motor bypass.
- Utility selection is disabled. Utility-head and intention-encoder weights are
  frozen. The existing context projection uses fixed column 1 in all tasks;
  it is no longer a selected behavior. Task requests enter through sensory inputs.
- **32 physical worlds**: 11 stand, 11 walk, 10 hover. Straight walking request
  1 cm/s; ground altitude request zero, hover 1.8–2.2 cm. Initial yaw ±0.15 rad.
- **180-second wall cap**, seed 71001, 16 CPU physics threads, RTX 4090 graph and
  optimization. Fresh Adam 1e-5, recurrent chunks of 32 actions (64 ms).
- Online imitation from the current physical state: execute **80% reference /
  20% student** commands and train on the reference's bounded motor targets.
  Equal mean loss per task; hover adds twice the wing-channel MSE. No utility loss.
- Reset only worlds leaving the height/upright training envelope or reaching
  two seconds. Save causal failure traces. Assistance and timeouts are not proof
  of autonomous success. Record setup, training, collection, optimization and
  actual transitions separately.
- Independently evaluate the saved checkpoint without teachers, mixtures,
  utility intervention or an external wing oscillator. Retain all three cases,
  including failures. Do not promote merely because imitation error decreases.

This is motor imitation, not PPO. The inherited walking reference and the
wing-motion hover reference generate training commands; only the student actor
controls acceptance evaluation. Later ground-outcome trials use explicit physical rewards (see below). The
legacy utility-PPO path rejects motor-only checkpoints; the separate
`--motor-ground` mode explicitly uses the motor objective.

The two-second implementation reference review kept all three tasks upright with
valid support and no numerical warnings. Hover root RMSE was 0.733 mm. Ground
raw heading gates still failed (stand/walk yaw RMSE 1.77/3.35 rad/s); these gates
remain unchanged. Reference control is not a learned result.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_flight_angle_probe_01.pt \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/motor_focus_01 \
  --device cuda --seconds 180 --worlds 32 --threads 16 --sequence 32 \
  --teacher-mix 0.8 --episode-seconds 2 --lr 0.00001 --seed 71001

uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus evaluate \
  --graph outputs/fly_survival/malecns \
  --resume outputs/embodied_fly/motor_focus_01/actor.pt \
  --output outputs/embodied_fly/motor_focus_01_evaluation \
  --device cuda --seconds 2 --seed 72001 --neural-view
```

Use unused output names when reproducing. Exit 2 means physical gates failed;
the complete evidence remains saved. New training videos are checked and opened
automatically for local review.

The fingerprint hashes compiled values exactly. Mac and Linux compilation differ
by rounding (maximum observed absolute difference2.60e-14, with all physical
arrays agreeing at1e-12). Each host's native/batched bodies match exactly; replay
uses the captured training MJB. Cross-host live recompilation must not bypass the
strict fingerprint check silently. This is not a different body preset.

## Measured outcome

The first [three-minute pilot](runs/motor_focus_01/SUMMARY.md) and its
[lower-assistance continuation](runs/motor_focus_02/SUMMARY.md) both failed
all three unassisted cases. Their combined training time was360.924648 s,
with324,608 real physical transitions in32 worlds. Assisted training survival
improved, but this has not become reliable stand/walk/hover control.

- [First complete review](../../previews/embodied_fly/motor_focus_01_all_tasks_v1.mp4).
- [Lower-assistance review](../../previews/embodied_fly/motor_focus_02_all_tasks_v1.mp4).

Both films include every case and its failed gate, actual simulated neural
activity and observer eye cameras. Earlier reviewed walking remains unchanged.

The [ground-wing correction](runs/motor_focus_03/SUMMARY.md) then adds the same
explicit wing-accuracy term to ground tasks. After181.184887 s of training,
unassisted standing and walking remain upright with valid foot support for two
seconds. A native five-second test confirms upright standing, with0.801 mm drift;
walking falls at3.28 s. Raw tracking gates and hover still fail. No runtime wing
override is used. These are motor-imitation results, not PPO results.

- [Updated complete three-task review](../../previews/embodied_fly/motor_focus_03_all_tasks_v1.mp4).
- [Full five-second ground cases](../../previews/embodied_fly/motor_focus_03_ground5s_v1.mp4).

## Initial form on the ground

`--ground-posture` teaches the initial pose during idle and restoring wing
commands during both idle and walking. The wing target depends on measured
angle and speed; it is no longer simply zero torque. Walking legs and hover
retain their task targets. All78 outputs remain learned at runtime.

Evaluation records every hinge's initial angle, separate leg/wing/body errors,
wing speed and body sag. These measured scores supplement the existing physical
gates. The motor_focus01–10 series optimizes corrective imitation; the later
ground_outcome series optimizes physical rewards. Passive joints are measured
but are not given invented actuators.

The first initial-form continuation04 regressed autonomous ground control under
25% reference assistance.05 and06 execute only student actions during collection
while fitting corrective labels. Both ground commands then remain upright for
five seconds;06 reduces standing sag to1.01% and initial leg/body RMS to
0.142/0.0587rad. Six wing joints still miss the stricter rest-angle gate.
Increasing ground wing weight10→100 in07 slightly improves body shape but worsens
wing-angle accuracy. Keep06 as the preferred development review; no promotion.

[Watch the current review](../../previews/embodied_fly/motor_focus_06_all_tasks_v1.mp4).

Reproduce its final continuation on the GPU host, after synchronizing the same
source and artifacts (use an unused output folder):

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_focus_05.pt \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/motor_focus_06_reproduction \
  --device cuda --seconds 120 --worlds 32 --threads 16 --sequence 32 \
  --teacher-mix 0 --ground-posture --ground-wing-loss 10 \
  --episode-seconds 2 --lr 0.00001 --seed 71006
```

The remaining gap is learned wing-position accuracy, not a missing physical
ability: the exact-pose reference passes. An unchanged continuation or another
larger loss weight is not yet justified by07. Preserve one body and one actor;
do not introduce runtime output masks to turn this into a visual-only success.

## Wing feedback diagnosis

The inherited walking observation format already contains every wing angle
and velocity. Dedicated continuous wing channels were added during the earlier
aerodynamic experiments to avoid clipping. Keeping these sensor channels does
not reintroduce aerodynamic forces: all current tasks retain `wing_motion`.

[Frozen response probes](runs/motor_response_01/SUMMARY.md) show weak restoring
responses and some wrong-sign commands. An opt-in training-only paired-response
loss teaches how the existing actor should react to small wing measurement
changes. [Pilot08](runs/motor_focus_08/SUMMARY.md) completes 181.01 s of training
without changing the body, observation scaling or runtime architecture.
Its response probe improves descriptively, but physical wing accuracy worsens.
Keep06 preferred. A useful response loss is not itself proof of motor success.

## Frozen decoder calibration

`ground_readout` tests whether existing MaleCNS motor features can decode the
required wing feedback. Collection executes the frozen parent actor on the
canonical body. Counterfactual angle/speed observations copy its actual prior
memory and do not advance physics. Only six existing output rows are fitted;
the graph, sensory encoding, hidden decoder, other output rows and physical
model stay fixed. Whole worlds, including all their variants, are held out.

The [first fit](runs/ground_readout_01/SUMMARY.md) lowers held-out command MSE
72.9% but regresses physical standing/walking. It is a diagnostic candidate,
not a new preferred policy. Previous-command sensitivity probes do not show
strong local self-amplification. A [follow-up fit](runs/ground_readout_02/SUMMARY.md)
pools the candidate's own physical histories with the original examples. It
recovers upright standing but loses walking stability; all full gates fail.
Keep06 preferred and stop output-row-only calibration. Future motor learning
must address the full sensory-to-motor representation and physical outcomes.

Reproduce the first fit on the GPU host with synchronized assets and unused
output directories; these are supervised collection/fitting commands, not PPO:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ground_readout collect \
  --checkpoint assets/embodied_fly/diagnostics/motor_focus_06.pt \
  --graph outputs/fly_survival/malecns --device cuda \
  --worlds 32 --threads 16 --seconds 2 --seed 73009 \
  --output outputs/embodied_fly/readout_reproduction_corpus
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ground_readout fit \
  --checkpoint assets/embodied_fly/diagnostics/motor_focus_06.pt \
  --cache outputs/embodied_fly/readout_reproduction_corpus --device cuda \
  --output outputs/embodied_fly/readout_reproduction
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus evaluate \
  --resume outputs/embodied_fly/readout_reproduction/actor.pt \
  --graph outputs/fly_survival/malecns --device cuda --seconds 5 --seed 72001 \
  --neural-view --output outputs/embodied_fly/readout_reproduction_evaluation
```

`ground_readout pool` accepts another cache with its original checkpoint only
after verifying that the complete upstream feature map is identical. It keeps
the source hashes and original training/validation world partitions. Pooling
reuses examples and collects no additional physical experience.

## Ground curriculum with physical disturbances

`motor_focus train --task-set ground` assigns16standing and16walking worlds
when using32worlds. It trains the existing sensory encoder, intrinsic cell
dynamics and motor decoder. Utility remains inactive. One resulting checkpoint
is still evaluated on allthree commands; hover remains unfinished and is not
trained during this ground stage.

`--wing-angle-perturbation` and `--wing-speed-perturbation` change only ground
training reset states. Angles stay within physical limits, and the full nominal
resting-pose target stays fixed. This creates actual recovery trajectories for
feedback learning. Evaluation retains the original undisturbed initial states.

The [first ground pilot](runs/motor_focus_09/SUMMARY.md) stays upright through
all128 completed training episodes and both five-second ground evaluations.
Wing accuracy still does not improve on06, so it is not promoted. The declared
follow-up changes only the optimizer learning rate in a continuation from09.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
  --resume assets/embodied_fly/diagnostics/motor_focus_06.pt \
  --graph outputs/fly_survival/malecns --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/ground_motor_reproduction --device cuda \
  --seconds 180 --worlds 32 --threads 16 --sequence 32 --lr 0.00001 \
  --teacher-mix 0 --task-set ground --ground-posture --ground-wing-loss 10 \
  --wing-response-loss 1 --wing-response-worlds 8 --episode-seconds 2 --seed 71009 \
  --wing-angle-perturbation 0.15 --wing-speed-perturbation 2
```

## Physical-outcome ground PPO

`embodied_fly.ppo --motor-ground --preset wing_motion` trains the motor-only
checkpoint directly on measured movement, support and posture. It requires the
parent's exact physical fingerprint and activates all78 motor outputs. No
utility loss, teacher or legacy rehearsal corpus is used. The critic and
exploration distribution exist only during training. The actor remains the
same command-conditioned MaleCNS network.

The ground curriculum rewards resting wing angles and low wing speed for both
standing and walking; standing additionally rewards the initial leg pose.
Body pose and height terms discourage collapse. Every reward is computed per
world per2ms action, rate-scaled by elapsed time. Full task evaluation remains
separate and includes hover, even when this curriculum stage only trains ground
commands. `motor_outcome.py` records the exact weights and scales in each run.

Reproduce the declared ground-outcome pilot with an unused output name. Execute
on the same platform as the parent checkpoint's compiled physical fingerprint.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-ground --preset wing_motion \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_focus_06.pt \
  --output outputs/embodied_fly/ground_outcome_reproduction \
  --device cuda --worlds 32 --threads 16 --seconds 180 \
  --horizon 128 --sequence 16 --epochs 2 --episode-seconds 2 \
  --gamma 0.998 --gae-lambda 0.99 --lr 0.00001 --noise 0.02 \
  --target-kl 0.03 --entropy 0.001 \
  --wing-angle-perturbation 0.15 --wing-speed-perturbation 2 --seed 81001

uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus evaluate \
  --graph outputs/fly_survival/malecns \
  --resume outputs/embodied_fly/ground_outcome_reproduction/actor.pt \
  --output outputs/embodied_fly/ground_outcome_reproduction_evaluation \
  --device cuda --seconds 5 --seed 72001 --ground-posture --neural-view
```

`--wing-supervision 100` adds an explicit auxiliary to motor-ground PPO: only
six wing commands are fitted to bounded angle/speed corrections from the current
physical state. Walking-leg and body commands still learn from physical rewards.
No teacher commands are executed or blended into control. The scalar multiplies
mean squared normalized-action error; it does not change a physical torque limit.
This variant is **PPO plus corrective supervision**, with separate gradient audits
and label-presentation counts. The auxiliary adds no deployed module or policy.

## Next hover investigation

The [clock audit](runs/hover_clock_audit_01/SUMMARY.md) shows that the current
hover reference can request different wing commands from identical current
sensor inputs. Recurrent history could supply phase, so this is not proof of
unlearnability. Before further hover training, test a reference driven by
measured wing position/speed and current altitude error. Its purpose is to
provide more direct teaching targets, not to become a deployed controller.

Keep the canonical body, torque limits and force law. Validate the reference
physically before training, including startup from zero wing speed. Then train
one continuing actor with ground rehearsal and hover examples together, using03
as the development parent. Do not change the actor into a scripted oscillator,
add a hidden phase input or substitute a controller during evaluation.
