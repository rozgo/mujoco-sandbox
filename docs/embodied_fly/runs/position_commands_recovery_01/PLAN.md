# Retain command switching while recovering hover

Declared before training, September 13, 2026. Position_commands01 now starts,
stops and resumes in all three continuous worlds, but fails yaw tracking and
loses hover. It is a useful motor candidate, not a replacement for the prior
combined baseline. Try one bounded recovery pilot from position_commands01.

Keep its same body, actor, fixed graph, 32-world distribution, one-second
ground command phases, five-second episodes, 32-action recurrent chunks and
ground supervision. Twelve switching worlds, five fixed stand, five fixed walk,
ten hover. All ground actions remain student actions and command changes do not
reset body state or neural memory. Fixed ground rehearsal now copies this parent.

Two deliberate training changes: reduce fresh Adam learning rate from 1e-4 to
1e-5, and execute 80% measured-state hover-reference / 20% student commands in
hover worlds only. The reference labels are the existing state-based wing
controller, with the same varied 1.8-2.2 cm starts. This supplies sustained-flight
examples while preserving learned ground transitions. It does not add an
oscillator, pose override, changed force law or teacher to deployed evaluation.
Explicitly record the assisted hover fraction in every training trace. Do not
count assisted survival as learned flight.

180-second training budget, seed 99403, 16 native CPU MuJoCo/mjbatch threads and
RTX 4090 neural work. Ground task weight 4, non-wing target weight 4, ground
wing loss 10; hover-response auxiliary loss remains zero. Utility stays disabled.

Evaluate five-second fresh-start stand/walk/hover at seed 97013, then the same
eight-second continuous ground sequence at seed 99103. Every evaluation action
is the actor's, with no masks, teachers or resets. Render all commands/worlds,
retain failures and unchanged gates. Do not select this checkpoint based on
imitation loss or training assistance. Hover robustness beyond the original
review and the full survival task remain separate requirements.
