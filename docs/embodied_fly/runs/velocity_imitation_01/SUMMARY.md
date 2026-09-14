# First minute of fresh velocity imitation

The fresh student does **not fly yet**. It fails the opening hover in both
student-only physical evaluations, dropping below 5 mm altitude at 0.070 seconds.
No translation or turning stage is reached. This is a preserved diagnostic
checkpoint, not a replacement for any accepted motor policy.

[Watch the 15-second, 1x PID/student comparison](../../../../previews/embodied_fly/velocity_imitation_01_pid_comparison_v1.mp4).
Both panels use the same physical model, initial state and command timestamps.
The student continues physically for 0.4 seconds after its first failure;
an explicit end card replaces its panel afterwards while the teacher continues.
No frozen body, repeated resets or assisted recovery is presented as learning.

## What ran

| Measurement | Observed result |
| --- | --- |
| Teacher collection | 10 CPU MuJoCo/mjbatch worlds, 10 threads |
| Teacher episodes | 8 training + 2 held-out, each one continuous 71.2 s exercise |
| Teacher checks | 50/50 in every episode; 500/500 across ten episodes |
| Unique physical data | 356,000 action transitions, 712,000 physics steps |
| Collection time | 219.386 s stepping/capture; 266.949 s total including setup, encoding and checksums |
| Fresh initialization | 4.879 s; no prior policy or optimizer loaded |
| Training hardware | RTX 4090; CPU physics is not running inside this replay optimizer |
| Training budget | 60 s, finishing the active optimizer update |
| Actual training | **62.555 s**, 14 Adam updates |
| Replay batch | 64 sequences × 128 supervised steps; 64 preceding warm-up steps |
| Supervised targets processed | 114,688; replay samples can overlap and are not new physical experience |
| Warm-up observations processed | 57,344; excluded from supervised loss |
| Unique training partition | 284,800 transitions; validation episodes never supply gradients |
| Stage coverage | All 50 stage IDs sampled in every update |
| Peak allocated CUDA memory | 7,528,224,256 bytes / 7.01 GiB |
| Network | 391 inputs, 78 outputs, 166,700 recurrent neurons, 25,582,938 fixed signed connections |
| Parameters | 2,515,378 total; 2,335,772 trainable |
| Physical clocks | 1,000 Hz physics, 500 Hz commands; four recurrent updates per command |
| Initial/final held-out wing MSE | 0.396961 → 0.008018 |
| Initial/final held-out posture MSE | 0.397086 → 0.211395 |
| Held-out validation time | 2.030 s before and 1.810 s after, outside training timer |
| Physical student evaluation | 0/2 complete exercises; 0.470 s captured per case |
| Physical evaluation loop time | 0.911 s and 0.785 s, excluding model/graph setup |
| Video rendering | 14.779 s; 750 frames at 50 fps, 1600×960 |

## What the loss improvement means

The 98% decrease in wing-command MSE is measured relative to random initialization.
It does **not** mean the student learned flight or the teacher's wingbeat.
A read-only audit compares the trained student with a constant predictor using
the training set's mean wing positions. That predictor achieves held-out wing
MSE 0.004135, better than the student's 0.008018. This analysis predictor is never
used to control the body.

Predicted sweep-command standard deviations are 0.0133 and 0.0180, versus target
values 0.1024 and 0.1013. Sweep correlations are only 0.121 and 0.261. During the
same first 70 ms of physical flight, the teacher's mean absolute sweep speeds
are 31.6/31.4 rad/s; the student's are 4.5/11.6 rad/s. These observations suggest
the first updates mostly fit mean joint positions, without learning the required
wing motion. They do not establish a single cause for the remaining failure.

The next useful result would be improvement beyond the constant-pose baseline
and sustained physical wing motion. A second minute can continue this checkpoint
with the same data, optimizer, sampler and recipe; no additional learning was
silently performed after this first burst.

## Reproduce and continue

Use the isolated locked Python environment. `MALECNS_GRAPH` below denotes the
local processed graph directory. Capture the demonstrations once:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_demonstrations \
  --contract-report docs/embodied_fly/runs/pid_velocity_fast_04/report.json \
  --output outputs/embodied_fly/velocity_teacher_dataset_01
uv run --project experiments/embodied_fly --locked python -m embodied_fly.fresh_velocity_actor \
  --graph "$MALECNS_GRAPH" --model outputs/embodied_fly/velocity_teacher_dataset_01/model.mjb \
  --output outputs/embodied_fly/fresh_velocity_actor_01 --seed 121101
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_imitation \
  --initial outputs/embodied_fly/fresh_velocity_actor_01/initial_actor.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_imitation_01 --seconds 60
```

Use EGL on headless Linux. Collect on the Linux host matching the accepted
physical fingerprint; do not weaken it to accommodate cross-platform compilation
differences. Mac rendering uses the transferred MJB and recorded states.

The continuation command below is prepared, **not executed**:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_imitation \
  --resume assets/embodied_fly/diagnostics/velocity_imitation_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_imitation_02 --seconds 60
```

Fresh and trained checkpoints, optimizer state, sampler state, source commits,
graph/model hashes, losses, failures and audit are preserved alongside this
report. Physics and training source were frozen during their respective runs.
Thirteen focused tests passed, including replay coverage, validation isolation,
sampler continuation and forward/gradient equivalence to the deployed actor.
