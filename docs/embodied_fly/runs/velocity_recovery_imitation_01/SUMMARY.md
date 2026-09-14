# Reordered recovery imitation: physical regression

This block does **not** improve flight. After another 30 minutes, all four
predeclared physical cases fail their opening hover: altitude drops below 5 mm
at 0.202 seconds for the three cold starts and 0.244 seconds for the held-out
recovery start. No movement stage is reached or passed. The original-start
result regresses from the parent's 5.624/5.626 seconds of airborne flight.
Preserve the parent as the better development reference; neither is a successful
flight controller. No intermediate checkpoint was substituted for the final one.

[Watch all four PID/student cases at 1x](../../../../previews/embodied_fly/velocity_recovery_imitation_01_pid_comparison_v1.mp4).
The student continues physically for 0.4 seconds after failure, then an explicit
end card replaces its panel. The matched teacher continues. The film is 27 s,
1,350 frames, 50 fps, 1600x960, and was fully decoded, inspected and opened.

```sh
open previews/embodied_fly/velocity_recovery_imitation_01_pid_comparison_v1.mp4
```

## What was implemented

- Ten distinct, seeded orders, retaining every one of the 50 original stages
  in each complete 71.2-second exercise. Movement/braking pairs stay together;
  inverse vertical pairs climb before descending to avoid deliberate floor
  collisions. There are no resets inside an exercise.
- PID recovery from saved student physical states. Ten two-second checks pass
  before full collection; the teacher stops the student's climbing/drifting
  starts while respecting the unchanged body, force law and joint controls.
  It initializes its own phase from measured wing angle/speed; phase remains
  outside the student's inputs. The live fly receives no pose/velocity correction.
- Combined original and new recordings: 16 training episodes and four held-out
  episodes. All 1,000 teacher stage checks pass. Sampling follows each episode's
  actual stage order and includes eight explicit episode starts per batch.
- Continuation of the same final checkpoint and Adam/RNG state, with an explicit
  dataset-change flag. Batch size, sequence length, learning rate, architecture,
  fixed graph, physics and loss remain unchanged.

The expected episode mixture is 50% original recordings, 12.5% reordered cold
starts and 37.5% reordered recoveries. This is recovery-state augmentation:
complete teacher episodes branch from saved student errors. It is **not** full
online DAgger with teacher corrections throughout newly collected student
rollouts. That distinction matters because most of each new recording again
contains successful teacher behavior.

## Measured result and diagnosis

| Measurement | Parent | New final checkpoint |
| --- | ---: | ---: |
| Wing MSE on the same new held-out windows | 0.0003313 | 0.0002880 |
| Non-wing MSE on those windows | 0.0003580 | 0.0001314 |
| Wing MSE on the original held-out audit windows | 0.0004835 | 0.0003887 |
| Sweep correlations on original held-out recordings | 0.989 / 0.990 | 0.990 / 0.990 |
| Pitch correlations on original held-out recordings | 0.642 / 0.670 | 0.722 / 0.781 |
| First altitude failure, original nominal / held-out | 5.624 / 5.626 s | 0.202 / 0.202 s |
| Complete physical exercises | 0/2 | 0/4 |

The final policy starts with approximately the right measured wing speed, then
its wing motion weakens under its own physical feedback. Mean absolute sweep
speed over 0.15–0.20 s is 17.6/17.0 rad/s, versus teacher 31.1/31.0 and parent
31.1/31.2. This explains the immediate loss of support as a measured failure
mechanism; it does not identify the sole learning cause. All four initial
observation vectors match their recorded teacher starts exactly, including the
previous-action history restored only at reset. The physical fingerprints match.

This trial shows that better command prediction on recorded inputs can coexist
with worse closed-loop flight. Longer training and these recovery snapshots did
not establish a self-sustaining wingbeat. Do not infer improvement from the
imitation-loss decrease or launch another unchanged multi-hour block.

The next investigation should test short self-sustained wing motion and teacher
corrections on the states/actions produced throughout actual student rollouts.
Use physical validation to establish an improvement before increasing training
time. No further training or method change was executed after this block.

## Timing and scale

| Measurement | Actual result |
| --- | --- |
| GPU | NVIDIA RTX 4090, PyTorch CUDA |
| Physics | CPU MuJoCo/mjbatch, 1,000 Hz; 500 Hz control |
| New full demonstration capture | 231.649219 s, 10 worlds, 356,000 actions |
| Dataset total process | 281.709578 s, including copying/compression/hashing |
| Recovery probe | 6.562624 s capture, 10.805237 s total; 10,000 new actions |
| Current learning block | 1,804.362681 s, 401 Adam updates |
| Cumulative learning ancestry | 3,670.512146 s / 61 min 10.5 s, 816 updates |
| Replay | 64 parallel sequences, 128 supervised + 64 context steps |
| New supervised replay targets | 3,284,992, not new physical transitions |
| Context slots | 1,642,496, including 205,312 masked pre-start padding slots |
| Unique training partition | 569,600 physical transitions |
| Physical worlds inside optimizer | 0; learning uses cached replay |
| Peak allocated CUDA memory | 8,197,219,328 bytes / 7.63 GiB |
| Learning setup | 9.531554 s |
| Before/after held-out validation | 2.037091 / 1.825127 s, outside learning timer |
| Checkpoint writes excluded from learning timer | 0.317734 s |
| Physical evaluation loops, episodes 0/8/10/18 | 1.190205 / 1.036721 / 1.037661 / 1.097139 s |
| Evaluation total including setup/captures | 14.033169 s |
| Frozen original-window motion audit | 6.716078 s |
| Video rendering / full decode | 31.318095 / 2.116871 s |

Learning ran 07:20:04.826103–07:50:11.492152 UTC on September 14, 2026 (end
includes final validation/write). Training revision 326524f, collection bcfb7cc,
evaluation/render 2fe364b. Same 391-input/78-output actor: 2,515,378 parameters,
2,335,772 trainable, 166,700 graph neurons and 25,582,938 fixed signed connections.
Loss is wing-command MSE + 0.1 times non-wing MSE; Adam 1e-4, gradient cap 1,
Float32, four recurrent updates/action. No PPO or critic runs in this block.
All six periodic snapshots plus the final optimizer checkpoint are archived.
The complete fly suite passed 218 tests; two additional random-command tests
passed separately. Existing sparse-tensor/NumPy deprecation warnings remain.

## Reproduction

Use the locked isolated environment and a graph directory in MALECNS_GRAPH.
On headless Linux set MUJOCO_GL=egl. Use new output paths to preserve evidence.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_imitation \
  --resume assets/embodied_fly/diagnostics/velocity_imitation_30m_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_recovery_dataset_01 \
  --output outputs/embodied_fly/velocity_recovery_imitation_01_reproduction \
  --seconds 1800 --cold-starts 8 --allow-dataset-change --snapshot-seconds 300
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_student \
  --checkpoint assets/embodied_fly/diagnostics/velocity_recovery_imitation_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_recovery_dataset_01 \
  --output outputs/embodied_fly/velocity_recovery_imitation_01_evaluation_reproduction \
  --episodes 0 8 10 18
```

The separately implemented [smooth random command generator](../../RANDOM_COMMANDS.md)
has reproducibility, continuity and range tests. It is not part of this dataset
or trained checkpoint; physical collection with altitude bounds remains future
work. An unlimited generator still requires finite, measured teacher simulation
and archived data/seed prefixes for every timed training run.
