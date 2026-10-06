# Surgical thread-insertion robot brief (v2)

October 6, 2026. Revised with the user after the first learned policy, to match
what we now know we can simulate, train and measure. The original brief is kept
as [BRIEF_V1.md](BRIEF_V1.md).

## Goal

A simulated robot that completes the full insertion cycle at several sites,
robustly, under realistic disturbances: pick a thread, align with a target while
avoiding vessels, insert to a prescribed depth, release, withdraw, and repeat.
Later: tissue surface motion at clinical amplitudes, multi-site sequences and
recovery from missed pickups. MuJoCo for physics, PufferLib 5.0 for learning.

## Principles

- **Learn every phase under disturbance.** A perfect simulated world makes any
  phase look easy to program. Real sensing noise, latency, actuator variation,
  vibration, tissue motion and airflow do not. Policies learn the mission; the
  programmed servo remains only as the motor drive, as on a real stage. Scripted
  controllers are yardsticks for measuring learning, not deployed components.
- **One disturbance layer for every task.** Each disturbance is a documented
  physical model with stated magnitudes, applied in the shared C core, so
  training, evaluation and replay see the same world. Magnitudes are
  illustrative until replaced by measured hardware and tissue data.
- **Measure the mission.** Success is judged on true simulator state: depth and
  placement error, vessel clearance, thread retention, peak tool and thread
  loads, and completed cycles. Report results on held-out seeds at stated
  disturbance levels, including full strength. Use more training seeds when
  comparing changes, not as a ritual gate.
- **Physics checks follow the task.** Numerical gates apply to the contact
  regimes the task uses. Calibration against the real thread, tissue and
  environment takes priority over further numerical refinement once targets are
  chosen.

## Stages

1. **Robust alignment** (current). Disturbance layer, then alignment under it,
   compared with the undisturbed policy and the scripted yardstick.
2. **Insertion into tissue.** A validated needle insertion model (puncture,
   insertion resistance, retention) on a moving tissue body; learn
   align-and-insert to depth while avoiding vessels.
3. **The full cycle with the full thread, offline.** Run pick, insert, release
   and withdraw end to end with the full elastic-rod thread to expose failure
   modes and define metrics; slow to simulate, but only needed a few times.
4. **A thread fast enough to learn with.** A reduced thread model validated
   against the full one on exactly the phases being learned, including contact
   and air drag, targeting about real time per environment.
5. **Learn the thread phases**, then multi-site sequences and recovery, and
   confirm on the full thread.

## Inputs needed

The real thread (material, cross-section, length), the tissue the phantom stands
for, and the hardware specifications (encoders, camera, stages, vibration
environment). Until then, presets and literature values are used and labelled.
