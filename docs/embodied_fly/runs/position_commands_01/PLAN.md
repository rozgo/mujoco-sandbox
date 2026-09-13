# Continuous ground command learning

Declared before training on September 13, 2026. The preserved motor parent
position_sustain_retention01 cannot start walking after two seconds of standing,
in any of the three verified continuous tests. Fresh-start primitive success
is insufficient for later utility selection. Train command changes within the
same physical episode and recurrent history.

One 180-second imitation pilot from that parent, seed 99303. The same canonical
wing_position body, force law, 397 inputs, 78 outputs, fixed MaleCNS wiring and
four graph updates remain. No utility or PPO training and no new runtime module.
All encoder/decoder and modeled cell parameters learn; normalization, graph,
utility and intention weights remain fixed. Fresh Adam 1e-4, chunks of 32 actions.

Use 32 worlds, 16 native CPU MuJoCo/mjbatch threads and RTX 4090 neural work.
Twelve ground worlds switch stand/walk every one second (six start in each
command). Five remain standing, five remain walking, ten rehearse hover. Five
seconds per episode. At command boundaries only the requested forward speed
and training task label change; body state and recurrent memory persist. Normal
failure/timeout resets remain explicit and logged, with initial commands restored.

The switching worlds receive current-state inherited walking-reference labels
or full initial-form stand targets. Copying the frozen parent would reinforce
its switching failure. Fixed ground worlds retain frozen-parent targets and
initial-form/resting-wing supervision. Hover retains the measured-state
reference. Ground task weight 4, ground non-wing target loss weight 4, ground
wing loss 10. The non-wing target source is explicitly mixed by world role;
the weight-4 term is not called frozen-parent retention for switching worlds.
All physical actions are student actions, with zero teacher blending. The
previous failed hover-response auxiliary loss is disabled.

Evaluate the same stand/walk/stop/resume sequence at seed 99103, two seconds per
phase, three worlds, without resets. Keep the predeclared gates unchanged and
render all three worlds, including failures. Also evaluate fresh-start stand,
walk and hover at seed 97013 for five seconds each and render all commands.
Preserve the baseline unless the evidence supports a practical improvement;
retain failures and report motor retention separately from switching performance.
