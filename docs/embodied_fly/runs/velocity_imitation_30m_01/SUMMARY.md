# Thirty-minute velocity imitation: wing motion improves, flight control fails

The final student stays airborne for 5.624/5.626 seconds on the two predeclared
starts, then falls. It climbs from 18.67 mm to about 99 mm and drifts, passing
none of the three completed stage checks in either case. Neither complete
71.2-second exercise succeeds. This is a preserved development checkpoint.

[Watch the 15.04-second, 1x PID/student comparison](../../../../previews/embodied_fly/velocity_imitation_30m_01_pid_comparison_v1.mp4).
Both cases, including the physical falls, are shown. Cameras follow with damping;
the small body-following inset helps show wings. The altitude/velocity charts
measure the recorded motion. Open locally with:

```sh
open previews/embodied_fly/velocity_imitation_30m_01_pid_comparison_v1.mp4
```

## What improved

| Measurement | First-minute parent | Final continuation |
| --- | ---: | ---: |
| Held-out wing-command MSE | 0.008018 | 0.000483 |
| Held-out non-wing posture MSE | 0.211395 | 0.000361 |
| Left/right sweep correlation on teacher recordings | 0.121 / 0.261 | 0.989 / 0.990 |
| Left/right predicted sweep standard deviation | 0.0133 / 0.0180 | 0.1008 / 0.1001 |
| Teacher target sweep standard deviation | 0.1024 / 0.1013 | Same recordings |
| Actual first-70-ms sweep speed, mean absolute, rad/s | 4.5 / 11.6 | 32.0 / 31.7 |
| Teacher actual first-70-ms sweep speed, rad/s | 31.6 / 31.4 | Same recordings |
| First altitude failure, nominal / held-out start | 0.070 / 0.070 s | 5.624 / 5.626 s |
| Complete physical exercises | 0/2 | 0/2 |

The constant training-mean wing-pose predictor has MSE 0.004135. The final
student beats that baseline by 88.3%, and its measured startup wing motion is
close to the teacher. This is progress beyond fitting a resting pose. However,
high correlation on recorded teacher inputs does not establish stable control
when the student's own actions determine its observations. Pitch-command
correlations remain 0.642/0.670, lower than sweep correlation. These observations
motivate better feedback training; they do not isolate a single failure cause.
The audit's 70-ms motion window is the same cold-start interval used for the
first-minute comparison, not the final student's failure time.

## Measured training and evaluation

| Measurement | Result |
| --- | --- |
| Learning device | NVIDIA RTX 4090, PyTorch CUDA |
| Physics | CPU MuJoCo/mjbatch; no live physics inside replay optimization |
| Source | Training 3917c7a; final evaluation/render 186e937 |
| Requested / actual learning | 1,800 s / 1,803.594666 s, finishing the active update |
| Parent + continuation learning | 1,866.149465 s / 31 min 6.15 s |
| New / cumulative Adam updates | 401 / 415 |
| Supervised replay targets processed | 3,284,992; overlapping samples are not new physical experience |
| Replay batch | 64 sequences, 128 supervised steps, 64 context slots |
| Cold-start sequences | 8 per batch, one per training episode; 3,208 total |
| Context slots | 1,642,496; includes 205,312 masked padding slots before time zero |
| Remaining sequences | 56 cover all 50 stage IDs every update |
| Unique training data | 284,800 transitions in eight complete exercises |
| Held-out data | Two complete exercises; never supply gradients |
| Clocks | 1,000 Hz physics, 500 Hz action/sensing; four core updates/action |
| Actor | 391 inputs, 78 outputs; 2,515,378 parameters, 2,335,772 trainable |
| Fixed graph | 166,700 neurons, 25,582,938 signed connections; hash unchanged |
| Peak allocated CUDA memory | 7,529,266,688 bytes / 7.01 GiB |
| Setup | 9.159773 s |
| Before/after validation | 2.030445 / 1.820664 s, outside learning timer |
| Periodic checkpoint writes | 0.320581 s excluded from learning timer |
| Final physical evaluation loops | 10.272602 / 10.151080 s |
| Final evaluation total including setup and saved captures | 30.308664 s |
| Read-only wing audit | 7.139563 s |
| Video rendering / full decode | 18.938412 / 1.264845 s |
| Video | 752 frames, 50 fps, 1600x960, 15.04 s, 1x |

This resumes the first-minute weights, Adam state and RNG, with the one
predeclared sampler change: eight explicit cold starts. Loss remains six-wing
MSE + 0.1 times other-joint MSE, Adam 1e-4, gradient cap 1, Float32. No PPO,
critic or old reward function runs. The final checkpoint is the primary result;
no intermediate was substituted for a better-looking outcome. All six periodic
snapshots and the final optimizer checkpoint are archived in Git LFS.

The ten original teacher episodes all contain the **same stage order**. Their
heading and wingbeat-phase variations do not constitute varied exercise order.
This is a gap in the demonstrations, now explicitly recorded.

## Midpoint peek and reporting incident

At the user's request, the saved 15-minute snapshot ran a one-second hover check
on CPU while GPU learning continued. It stayed upright but climbed to 35.20 mm.
Setup took 9.101697 s and evaluation 113.710033 s on two Torch threads. This was
not a full exercise or a checkpoint-selection test. Concurrent CPU work may have
affected the shared host's throughput; the main learning wall clock continued.

The first final-evaluation attempt saved its nominal trajectory but failed to
serialize a NumPy scalar in the stage report. Commit 186e937 fixes the conversion
and adds a serialization assertion. Both predefined physical cases were repeated
without changing weights or physics. The aborted trace/log remain in outputs;
its lost evaluation timer is unknown, not included in the successful timer above.
No additional neural training occurred. All 214 fly tests pass; existing upstream
sparse-tensor and NumPy deprecation warnings remain.

## Reproduce

Use the previously collected dataset and processed graph; do not overwrite output
folders. On headless Linux set MUJOCO_GL=egl. The learning source was 3917c7a;
use 186e937 or later for the fixed report writer.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_imitation \
  --resume assets/embodied_fly/diagnostics/velocity_imitation_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_imitation_30m_01_reproduction \
  --seconds 1800 --cold-starts 8 --allow-sampling-change --snapshot-seconds 300
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_student \
  --checkpoint assets/embodied_fly/diagnostics/velocity_imitation_30m_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_imitation_30m_01_evaluation_reproduction
```

Exact wall-limited update counts may differ with host load. Training metadata,
checkpoints, validation, trajectories and video hashes preserve the actual run.

The user has approved a follow-up: vary ordering while retaining every skill in
every complete exercise; collect PID corrections on student-visited states;
continue this same checkpoint and judge progress by physical flight. Longer
training on the unchanged recordings is not the chosen next step.
