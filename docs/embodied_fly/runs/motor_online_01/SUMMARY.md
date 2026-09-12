# Online braking correction 01

One MaleCNS-based actor can now retain near-target normal walking, brake and
resume without clearing physical or recurrent state. This candidate still has a
slow-walk fall and inaccurate heading, so it is **not an accepted complete
locomotion policy**. Flight and survival are not learned by this checkpoint.

| Measured training statistic | Result |
| --- | ---: |
| Training loop wall time | 300.229040 s |
| Setup | 9.815098 s |
| Collection and supervised forward computation | 225.503499 s |
| Backward and optimization | 74.220068 s |
| Simultaneous physical worlds | 32 |
| CPU physics threads | 16 |
| Physical transitions | 321,280 |
| Aggregate simulated time | 642.56 s |
| Optimizer updates | 1,255 |
| Command switches without resets | 159 |
| Peak allocated CUDA memory | 2,995,016,704 bytes |
| Completed training episodes / falls | 324 / 26 |

Physics is native MuJoCo CPU through mjbatch, at 5 kHz; the actor runs at 500 Hz.
The RTX 4090 executes full-graph inference and learning. This is **online
supervised imitation**, not PPO: inherited-teacher braking targets at zero command,
frozen PPO01 graph-actor retention targets while moving, fourfold moving loss
weight, and 0.02 rest/explore utility cross-entropy. Training assistance starts at
50% and fades to zero halfway through the timed run. No teacher or frozen actor
is deployed in any reported student evaluation.

The same 2,409,132-parameter actor retains the measured 166,700-neuron /
25,582,938-connection graph. Its internal excitability, leak and bias parameters
received finite gradients and changed. Four graph updates occur per control tick;
commands and utility intentions enter this recurrent controller before its 78
motor outputs. The walking-stage mask keeps 19 passive appendage channels at raw
zero, leaving 59 actively controlled outputs without removing anatomy.

## Teacher-free physical result

| Continuous phase | Forward target | Mean forward speed after settling | Additional result |
| --- | ---: | ---: | --- |
| Walk, 2 s | 1 cm/s | 0.951 cm/s | Upright, permitted support |
| Stop, 2 s | 0 | −0.000105 cm/s | 1.113 mm total travel; 0.0609 mm late drift |
| Resume, 2 s | 1 cm/s | 0.922 cm/s | Upright, permitted support |

No body or memory reset occurred at either command boundary. Mean yaw while
walking/resuming was 0.372 / 0.389 rad/s, exceeding the declared 0.35 gate. Stop
raw planar RMS was 0.134 cm/s, above the 0.1 raw gate despite very small drift.
The overall transition test therefore remains failed. These are development
measurements, not a final held-out robustness evaluation.

Five of six separate fixed-command cases stayed upright with permitted support.
The slow case toppled and loaded prohibited body parts, reaching 13.13 body weights
of peak prohibited normal force. All six original raw tracking gates failed.
Normal walking's 100 ms forward RMSE improved from PPO01's approximately 0.124 to
0.060 cm/s. That single metric does not outweigh the remaining failure. All tests
had zero MuJoCo numerical warnings; physical falls are distinct from solver failure.

Every saved training-fall trace was hash-verified and finite. The complete review
film includes all six cases and the full continuous sequence, including failures.
It is a development video, not the final interacting-fly demo.

## Provenance and reproduction

- Training source: `34981d42315c0b7746a7780344a78adf9f2c2b4c`.
- Student checkpoint: `9b9ab703483a52d06eb1700172e762b8226e3b4b6597678176a89bd886b6649a`.
- Parent: PPO01, `6983be3ccc477ecab3812e751c26e77311f93390c94830821ba8037aedb3449f`.
- The failed PPO02 and one-minute online pilot are separate experiments, not
  this checkpoint's ancestry. This actor inherits 600.083779 seconds of our
  supervised optimization plus 301.196457 seconds of our PPO training before
  the present stage. The inherited teacher's prior training is excluded.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.online \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_ppo_01.pt \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/motor_online_01_reproduction \
  --worlds 32 --threads 16 --seconds 300 --lr 0.00003 \
  --retention-weight 4 --seed 38002
```

EGL and CUDA training target the Linux GPU machine. Use the actual graph location;
Mac rendering uses its native backend without `MUJOCO_GL=egl`. Timed runs complete
their last recurrent chunk before saving, so measured wall time exceeds the
requested bound slightly and update counts can vary with machine load.
