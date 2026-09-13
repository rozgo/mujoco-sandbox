# Moving examples and early hover correction

Continuation begins Sep 13, 2026 at 13:55:50 UTC. Previous turn made progress:
versioned wing-position actuation, exact migration, two training pilots and a
complete standing pass. Walking and hover remain incomplete.

The retained walking actor supplies little forward movement. A native reference
probe on the same wing_position body compares receding and initial-path previews.
At 1 cm/s the receding case moves -0.453 cm along its initial heading; the
world-path case moves 5.001 cm in five seconds with valid support. At 2 cm/s,
world-path advances 10.010 cm. Yaw RMS remains high; these are not accepted motor
policies. An initial probe stopped on NumPy-bool JSON serialization after one
physical rollout; that failed output is preserved. Probe02 fixes serialization.

The batched anchored teacher caps planar path error at 0.15 cm, then supplies
the inherited 130 ms preview. Current body state is measured. Future path and
initial heading are privileged training inputs to the teacher only; the deployed
actor still receives its unchanged causal 397-input schema. This does not establish
that the student can reproduce the reference or infer unobserved absolute error.
The anchored five-second reference has 0.185 mm walking root RMSE and valid
support, but yaw RMS 3.091 rad/s fails the original gate. Stand/hover references pass.

The prior autonomous hover reverses its sweep command at 2 ms: actor -0.356,
current-state reference +0.217. Its first command at rest is near the reference,
so initialization alone is not sufficient. The next training pilot emphasizes
errors in the first 50 ms of hover by 10x. This weights existing samples; it adds
no physical transitions, actor input, runtime controller or artificial phase.

Pilot: resume position_fullbody01; 180 seconds; 32 worlds (11 stand, 11 walk,
10 hover), 16 CPU physics threads, RTX 4090 neural work, 5 kHz physics / 500 Hz
control, 32-action chunks, two-second training episodes, fresh Adam .0001.
Train all motor/cell-dynamics parameters, keeping graph edges and utility/intentions
fixed. Stand target is the complete initial actuator pose. Walking collection
uses 100% anchored teacher, hover and standing execute 100% student. Walking
has actual moving examples; hover corrective labels now come from its own
physical states. Task weights remain 4:4:1; wing weights remain 10 ground / 2 hover.
No retained-actor body loss or legacy dataset. Training seed 96003.

Evaluate seed 96013 for five complete seconds per command, one checkpoint,
no teachers/masks/resets. Preserve all physical/posture gates and failed captures.
Record actual time and every failure; produce and open the full 15-second 1x
review. Keep the standing candidate and older videos intact. This pilot is motor
imitation, not PPO and not the finished utility/survival simulator.
