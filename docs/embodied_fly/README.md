# Embodied fly: learned utility and direct body control

Work in progress on `feature/fly-brain`. This is the approved successor to the
utility-over-supplied-gait prototype. Full learned walking, flight, survival and
multi-agent behavior have not yet passed acceptance.

The current [ground review](../../previews/embodied_fly/student_angle01_walk_stop_walk_v1.mp4)
shows one graph actor walking, attempting to stop and resuming, with synchronized
simulated neural activity and observer eyes. Its stop gate still fails. Later
wing-only fits remain diagnostics: improved expert-history predictions have not
produced stable unassisted flight. [Startup trials](runs/motor_wing_startup_02/SUMMARY.md)
and the [matched neural-memory handover](runs/flight_handover_01/SUMMARY.md) explain
the latest evidence. The inherited flight expert remains a separate reference.

The subsequent [nonlinear motor-decoder stage](MOTOR_DECODER.md) preserves the
deployed network shape and learns from actual motor-cell states. Its supervised
fit and short physical-reward continuation still fail flight; their full ground
and flight reports are retained. The reviewed ground film remains unchanged.

The [learning journal](LEARNING_JOURNAL.md) records the hypotheses, failures,
architecture changes, measured runs and reporting limits. Exact GPU reports and
curves are in [`runs/`](runs/). The first five-minute graph warm start reduced
offline imitation error, but failed all six teacher-free physical acceptance cases;
its checkpoint is retained as diagnostic.

The subsequent walking-stage actuator mask kept six of six cases upright with
valid foot support. Stop/turn tracking is still under evaluation. Earlier reports
used rotated inertia axes for velocity metrics; corrected reports explicitly use
anatomical thorax axes. The same actor's original six velocity observation features
retain their inertia-frame coordinates for checkpoint compatibility.

The actor contains a measured MaleCNS graph with learned internal cell dynamics,
a learned utility head and an individual-actuator decoder. Utilities choose among
rest, exploration, feeding, drinking, escape and grooming; names declare intended
curriculum activities, not already acquired skills. The selected intention feeds
back into the recurrent core. It never invokes a Python walking routine.

All flies will share one checkpoint and have independent recurrent state. The
graph has 166,700 nodes and 25,582,938 aggregated neuron-pair connections. Its
normalized signed weights are fixed; cell excitability, leak and bias, sensory
encoding, intention encoding and motor decoding are trainable. State is latent
activity, not measured membrane voltage or reconstructed biological spikes.
Predicted transmitters inform sign assumptions; physiology and interfaces remain
engineering choices. Training connects body feedback to annotated sensory pools,
reads utilities from descending-neuron activity, and reads actuators from motor
neuron activity. There is no raw-observation-to-motor bypass.

The complete FlyBody has 108 degrees of freedom and 78 bounded actuator channels,
including legs, wings, mouth and antennae. Mass is approximately 0.985 mg. It is a
free body with foot adhesion, contact, passive tissue terms and bounded actuation.
The upstream CGS convention is retained to preserve teacher/physics compatibility:
cm, g, s; public metrics convert to SI. Torque units convert with 1 dyne·cm =
1e-7 N·m, force with 1 dyne = 1e-5 N. Exact joint and actuator limits are generated
beside the [static preview](../../previews/embodied_fly/anatomy_v1.png).
The initial walking curriculum uses 5 kHz native MuJoCo physics and 500 Hz control.
The flight preset uses 20 kHz physics / 5 kHz wing control. The inherited expert
can hover and fly forward with the complete body; the first shared-actor wing
imitation pilot failed physical flight and regressed ground behavior.

## Current implementation checks

- Sparse-core values and gradients match a dense small-graph reference.
- Motor/utility loss produces finite, nonzero internal-core gradients and changes
  internal parameters. This unit test is not full-graph learning evidence.
- Recurrent state persists across decisions; resetting one world preserves others.
- Disconnecting topology prevents observations reaching motor outputs.
- Complete static anatomy compiles and renders on Apple Silicon.
- The upstream learned walking teacher has been exported from its official
  TensorFlow SavedModel, with retained numerical reference samples. Four 0.5-second
  physical probes on the complete body held/walked at 0, 0.5, 1 and 2 cm/s without
  numerical warnings. Those inherited teacher skills are not our training result.

## Reproduce setup and teacher export

```sh
uv sync --project experiments/embodied_fly --locked
uv run --project experiments/embodied_fly --locked python -m embodied_fly.body \
  --preview previews/embodied_fly/anatomy_v1.png
uv run --project experiments/embodied_fly --locked pytest experiments/embodied_fly/tests -q
```

New collection/training/evaluation output directories must have new names. These
commands refuse to overwrite previous runs. Each new report automatically records
UTC timestamps, source commit, package source hashes, dependency versions and
whether there were uncommitted package changes. Model/video/state hashes tie a
presentation to its actual inputs; generation, training and evaluation times stay
separate. Earlier reports retain the evidence available when they ran.

Use uv 0.12.12 or compatible. The current MuJoCo 3.13 / PyTorch 2.14 stack uses
Python 3.12 because dm-control 1.0.46 imports labmaze's compiled extension, whose
latest release has no Mac wheels for newer Python minors. Python 3.14 with that
dependency excluded failed at import; it is not a supported configuration here.
FlyBody pins NumPy 1.26 upstream; this isolated project overrides it to 2.5.3 and
tests the actual paths used. Some upstream NumPy deprecation warnings remain.

Official teachers are in the FlyBody Figshare `trained-fly-policies.zip`, file
44815195, supplied MD5 `12934d5a1c60631a710bc2b6d297d3ce`, SHA-256
`2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582`.
TensorFlow 2.21 teacher export runs separately on Python 3.13. The export script
registers two renamed TFP SavedModel type aliases only inside that process;
PyTorch runtime does not load TensorFlow or TFP.

```sh
uv run --project experiments/embodied_fly/teacher_export --locked python \
  experiments/embodied_fly/teacher_export/export.py \
  outputs/embodied_fly/data/walking outputs/embodied_fly/data/walking_teacher.npz
uv run --project experiments/embodied_fly --locked python -m embodied_fly.collect \
  --teacher outputs/embodied_fly/data/walking_teacher.npz \
  --output outputs/embodied_fly/demonstrations_01 --seconds 2 --episodes 16
```

Teacher demonstrations include privileged future COM targets. Student observations
contain current joint position/velocity, activation, local body velocity/orientation,
foot contact, previous action, current command and needs. They contain no future
reference, gait phase, clock or ghost position. Dataset collection, supervised
learning, later RL, evaluation and rendering will be timed separately.

## Continue the motor curriculum

These are the selected short-pilot settings, with independent data, optimization
and evaluation clocks. `--worlds 32` means simultaneous neural sequences during
offline imitation. Data collection runs one native CPU physics world with CUDA
brain inference. The initial curriculum keeps 19 nonwalking actuator commands at
raw zero; it retains the complete body and all 78 actor outputs.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.dagger \
  --checkpoint assets/embodied_fly/diagnostics/motor_bc_01.pt \
  --graph outputs/fly_survival/malecns \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/dagger_01 --episodes 16 --seconds 2 \
  --student-fraction 0.25
uv run --project experiments/embodied_fly --locked python -m embodied_fly.train \
  --graph outputs/fly_survival/malecns \
  --data outputs/embodied_fly/demonstrations_02 \
  --additional-data outputs/embodied_fly/dagger_01 \
  --resume assets/embodied_fly/diagnostics/motor_bc_01.pt \
  --output outputs/embodied_fly/motor_dagger_01 \
  --worlds 32 --seconds 300 --reset-fraction 0.25 --seed 18002
uv run --project experiments/embodied_fly --locked python -m embodied_fly.evaluate \
  --checkpoint outputs/embodied_fly/motor_dagger_01/actor.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/motor_dagger_01_evaluation --walking-action-mask
```

Old checkpoints lack optimizer state; their first continuation resets Adam while
retaining weights and observation normalization. New checkpoints also retain Adam
state and parent hashes. Dataset aggregation remains supervised learning, not PPO.
Recorded demonstrations and teacher-free student rollouts can be inspected with
`python -m embodied_fly.tracking CAPTURE NEW_REPORT.json` for anatomical velocity
and block-mean tracking diagnostics, without changing acceptance gates or physics.

Add `--neural-view` to evaluation to capture a 128 × 128 anatomical activity map
at 50 Hz. The optional observer path bins all **140,638 spatially located cells**
using measured MaleCNS X/Z coordinates. The other **26,062 cells remain in the
brain computation**. Each pixel retains mean signed and mean absolute latent
activity; fixed amber/teal colors indicate its sign, not measured excitation or
biological spikes. Binning runs on the actor device, then only the small map moves
to the recorder. It does not affect actions, rewards or recurrent memory.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.record \
  outputs/embodied_fly/motor_dagger_01_evaluation walk \
  previews/embodied_fly/student_dagger01_walk.mp4
```

Use a new output name for every recording. Neural maps appear automatically when
the source capture contains them; older diagnostic captures remain usable.

## Source and attribution

- [Accepted curriculum and research](../fly_survival/COURSE_CORRECTION.md).
- [MaleCNS v1.0 official data](https://male-cns.janelia.org/download/), CC BY 4.0.
  Existing checksummed graph cache is reused from the survival prototype.
- [FlyBody](https://github.com/TuragaLab/flybody), Apache-2.0,
  pinned `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
- [Official FlyBody data/teachers](https://figshare.com/articles/dataset/25309105),
  **GPL-3.0-or-later** according to the official Figshare record. Converted teacher
  weights, attribution, original source link and license text are retained under
  `assets/embodied_fly/teachers/`; this is separate from the Apache-2.0 body code.
- [FlyBody paper](https://www.nature.com/articles/s41586-025-09029-4).
- [FlyGM](https://arxiv.org/html/2602.17997v3) informs the trainable internal-core
  approach; our implementation is not its released reproduction or its FlyWire data.

## Physical-outcome learning

The new `embodied_fly.ppo` path runs **32 independent complete FlyBody worlds**
through native MuJoCo CPU threads, with the shared full MaleCNS actor on CUDA.
These are separate training worlds, not yet interacting flies in the survival
arena. The retained walking actor has 383 inputs, four recurrent graph updates and
78 bounded outputs; the walking stage keeps 19 nonwalking channels passive.
Flight pilot 02 adds six continuous wing-speed inputs through the same sensory
encoder (389 inputs). Its checkpoint also works with the native batched learning
path. It is a diagnostic candidate, not a replacement for the walking reference.
A separate 1,697 → 128 → 128 → 1 critic is used only during learning.

The [declared recipe and first measurements](LEARNING_JOURNAL.md) distinguish
physical collection, PPO optimization, explicit imitation rehearsal and setup.
The 21.075-second implementation pilot collected 30,720 transitions, approximately
1,458/s including learning. It maintained upright valid-foot support in six
teacher-free tests, but did not solve stopping or all command-tracking gates.
Physical-outcome gradients reached trainable internal cells; useful neural
causality and the full locomotion/survival goal remain to be established.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_dagger_01.pt \
  --rehearsal outputs/embodied_fly/demonstrations_02 \
  --output outputs/embodied_fly/motor_ppo_01 \
  --worlds 32 --threads 16 --seconds 300 --lr 0.000002 --seed 28002
```

The requested duration bounds the training loop; its final rollout completes
before saving, and actual wall time is reported. No teacher actions are executed
in live PPO physics. The separate rehearsal batch uses only the original
whole-episode training split. Rates reward physical tracking and support every
2 ms; terminal falls get one additional penalty. Timeouts bootstrap their actual
final state and reset only the affected world's recurrent memory. New runs retain
short physical traces of training falls and a compiled model; the initial short
pilot retained fall metrics only.

Evaluation saves raw tracking gates plus 100 ms block-mean diagnostics. A CLI
exit status of **2** means the declared physical task gates failed, while saved
reports and trajectories remain available for inspection. Default seed 80001 is
explicitly a repeatedly used development/model-selection seed; it is not a final
held-out robustness test. `--seed` permits separate final evaluation seeds later.
Neither filtered diagnostics nor readable walking footage silently changes a
failed gate. Subsequent curricula must also validate commands changing within an
episode, before adding flight, needs and the shared survival arena.

Native batching reuses [mjbatch](https://github.com/kevinzakka/mjbatch) with the
repository's pinned source and Apache-2.0 attribution. Additional contact sensors
follow the [MuJoCo contact sensor specification](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact);
a regression test compares this sensor/batch path with the original physical body.

The [PPO 01 report](runs/motor_ppo_01/SUMMARY.md) records the actual five-minute
trial, including its mixed tracking outcome and failed stopping test. The
[complete-case review video](../../previews/embodied_fly/student_ppo01_all_commands_v2.mp4)
shows all six two-second cases from that single checkpoint, including failures.
It is a development film; flight and the interacting survival arena remain open.

```sh
open previews/embodied_fly/student_ppo01_all_commands_v2.mp4
uv run --project experiments/embodied_fly --locked python -m embodied_fly.montage \
  outputs/embodied_fly/motor_ppo_01_evaluation \
  previews/embodied_fly/student_ppo01_all_commands_v2.mp4
```

The reproduction command requires the captured states and an unused output name.
`evaluate --diagnostic-activity rest --cases 1` is an explicitly forced utility
intervention to distinguish selection from motor response. It cannot establish
acceptance of the deployed policy. The PPO 01 intervention still moved, identifying
motor stopping as an unresolved requirement.

The [online braking report](runs/motor_online_01/SUMMARY.md) records the next
five-minute supervised correction. Its one student checkpoint walks, stops and
resumes continuously, with 0.0609 mm of late stop drift, but still fails slow-walk
stability and heading gates. The [18-second complete review](../../previews/embodied_fly/student_online01_all_commands_and_transitions.mp4)
includes all six fixed-command cases and the uninterrupted transition, including
the slow-walk fall. It remains a development film, with actual recurrent activity
and observer eye cameras; no teacher executes actions in these captures.

The [flight curriculum report](FLIGHT_PREREQUISITES.md) records the working
inherited flight expert, failed clock transfer, and first mixed walking/wing
learning pilot. A [one-second reference hover](../../previews/embodied_fly/flight_teacher_hover_v1.mp4)
shows teacher-controlled flight, explicitly labeled with its upstream wingbeat
generator. The student has not learned reliable flight. Its lower offline error
did not translate to physical success; the retained walking reference remains
online01. Full survival behavior and multi-agent integration are still open.

The [continuous-wing feedback pilot](runs/motor_flight_probe_02/SUMMARY.md) and
[corrective flight pilot](runs/motor_flight_correction_probe_01/SUMMARY.md) retain
their measured training costs and failed physical outcomes. Eight assisted
correction episodes stayed airborne; unassisted student flight still failed.
These candidates use the same 389-input graph architecture and remain diagnostic.
The subsequent angle-feedback candidate expands that same actor to **395 inputs**
and retains stable ground support across all six development cases. Its
[six-second walk/stop/resume review](../../previews/embodied_fly/student_angle01_walk_stop_walk_v1.mp4)
shows the shared controller, actual simulated neural state and observer eyes.
Walk/resume phase gates pass; stopping still drifts. Both flight tests still fail,
and faster sensory learning regressed ground stability. The inherited flight
teacher and preserved student walking reference remain separately labeled.

```sh
open previews/embodied_fly/student_angle01_walk_stop_walk_v1.mp4
```

Batched airborne PPO uses the same graph actor and complete flight physics.
The reset corpus supplies only the first physical frame of training episodes;
no teacher controls live rollouts. Ground rehearsal remains explicitly measured.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_flight_angle_probe_01.pt \
  --rehearsal outputs/embodied_fly/retention_online01_01 \
  --flight-resets outputs/embodied_fly/flight_demonstrations_01 --preset flight \
  --output outputs/embodied_fly/motor_flight_ppo_probe_01 \
  --worlds 32 --threads 16 --seconds 60 --horizon 128 --sequence 32 \
  --epochs 2 --episode-seconds 0.06 --lr 0.000005 --noise 0.04 --seed 50001
```

That [PPO pilot](runs/motor_flight_ppo_probe_01/SUMMARY.md) failed flight and is
retained. A subsequent [wing-output calibration](runs/motor_wing_readout_01/SUMMARY.md)
updates only six rows/biases in the existing motor layer. Every other actor
parameter stays identical. It improves prediction on expert histories and
increases measured aerodynamic force, but still fails physical flight. Neither
candidate is an accepted flight policy.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.wing_observability \
  --graph outputs/fly_survival/malecns \
  --checkpoint assets/embodied_fly/diagnostics/motor_flight_angle_probe_01.pt \
  --data outputs/embodied_fly/flight_demonstrations_01 \
  --output outputs/embodied_fly/wing_observability_reproduction.json \
  --cache outputs/embodied_fly/wing_hidden_reproduction.npz
uv run --project experiments/embodied_fly --locked python -m embodied_fly.wing_readout \
  --resume assets/embodied_fly/diagnostics/motor_flight_angle_probe_01.pt \
  --cache-report outputs/embodied_fly/wing_observability_reproduction.json \
  --cache outputs/embodied_fly/wing_hidden_reproduction.npz \
  --output outputs/embodied_fly/wing_readout_reproduction \
  --seconds 10 --lr 0.003 --batch-size 1024 --seed 51001
```

The cached-feature fit has zero live physics worlds and introduces no extra
deployed brain. The existing graph, utility and full motor decoder still execute
at runtime. Its raw evaluations remain essential because low imitation error
does not establish a stable closed-loop wingbeat.

The current course correction is [flight dynamics driven by wing motion](WING_MOTION.md):
keep the same brain and six wing controls; a custom force law reads actual
wing motion and drives the free body. This is the current flight development
path. Earlier aerodynamic trials remain available as separate diagnostics.
