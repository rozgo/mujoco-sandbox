# Embodied fly: learned utility and direct body control

Work in progress on `feature/fly-brain`. This is the approved successor to the
utility-over-supplied-gait prototype. Full learned walking, flight, survival and
multi-agent behavior have not yet passed acceptance.

The [learning journal](LEARNING_JOURNAL.md) records the hypotheses, failures,
architecture changes, measured runs and reporting limits. Exact GPU reports and
curves are in [`runs/`](runs/). The first five-minute graph warm start reduced
offline imitation error, but failed all six teacher-free physical acceptance cases;
its checkpoint is retained as diagnostic.

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
Wing flight parameters and a higher-rate flight curriculum still need validation.

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
