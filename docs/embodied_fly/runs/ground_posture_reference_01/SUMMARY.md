# Ground posture reference

User review identified body crouching and wings folding into the body. This
reference proves the unchanged physical body can hold its canonical initial
shape using bounded existing actuators. It is not a learned result.

- Stand: all position-controlled joints target initial angles, foot adhesion 0.
- Stand and walk: six wing torques restore initial angles and zero velocity.
- Walking legs and hover reference remain unchanged.
- 3 worlds×2 seconds; setup 3.312887 s, capture 4.118766 s, no numerical warnings.
- Stand and hover pass full gates. Walk has valid upright support and rest wings,
  but fails pre-existing speed/heading gates.
- Stand group RMS: legs 0.01199 rad, wings 0, remaining body 0.04343 rad; zero sag.

The preceding two-world constant-pose check compares zero and half foot adhesion.
Both hold the body; choose zero. No training, added constraints or body changes.

Training uses corrective imitation targets, not gradients through the physical
posture measurements. All 102 hinges are measured, including passive joints.
