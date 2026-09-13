# Height and vertical-speed feedback imitation pilot

Declared before training, September 13, 2026. Parent remains
position_sustain_retention_01. The frozen-history probe finds much weaker local
altitude response than the training reference, and wrong-sign vertical-speed
responses at some sampled states. This is a local diagnosis, not proof of the
complete cause of oscillation.

Resume the existing unassisted motor imitation recipe: 32 worlds (11 stand,
11 walk, 10 hover), 16 CPU MuJoCo/mjbatch threads, RTX 4090 neural work, 180-second
training budget, 32-action recurrent chunks, fresh Adam 1e-4, five-second
episodes. Seed 99203. Every physical action is the student action. The same
body, force law, 397-input/78-output actor, fixed graph and utility freeze apply.
Ground task weighting 4, non-wing retention weight 4, ground wing loss 10,
initial-form standing, current-state hover labels and frozen-parent walking
labels are unchanged. No command-transition training is added in this pilot.

Add one training-only paired-response loss per chunk, weight 0.01. Sample eight
hovering worlds and clone their actual preceding neural memory. For each, vary
either measured altitude by +/-0.2 cm or world vertical speed by +/-2 cm/s.
Keep previous actions and all unrelated inputs unchanged. Fit the difference
between the two predicted wing commands to the reference's difference, divided
by fixed scales 0.02 (height) or 0.01 (speed). These 16 synthetic sensor inputs
per update are counted separately and are not additional physical experience.
The physical world and its neural history are not advanced by the probes.

First repeat the unchanged five-second stand/walk/hover review at seed 97013;
retain all outcomes and render every case. If airborne control survives, check
the same three additional starts used in the PPO diagnosis. Reduced imitation
loss alone will not promote a checkpoint. Keep the previous baseline available;
standing/walking, hover accuracy, bobbing and starting-state robustness must all
be reported. Continuous command switching remains a separate known gap.
