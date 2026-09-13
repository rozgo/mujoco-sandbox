# Learn full standing posture and supported flight examples

Follow-up to position_feedback01, after its complete unassisted review. The
position actuators give accurate resting wings, but the retained ground brain
still crouches/drifts and autonomous hover fails. Preserve that checkpoint/video.

Correct the training targets: standing receives the complete original actuator
pose, as the user requested. Walking retains the frozen parent's body commands.
Previously --retain-ground replaced standing body labels with the inherited
controller's outputs; that could preserve a crouch instead of teaching the rest
pose. The new --stand-initial-form flag makes this distinction explicit, only in
training supervision. There are no runtime output substitutions.

Train the full existing actor and modeled neuron dynamics, with the measured
connectome edges, utility head and intention projection fixed. The prior wing-only
parameter restriction cannot directly repair the other motor outputs. There is
still one actor, 397 inputs / 78 outputs, and one unchanged wing_position body.

A predeclared 180 s pilot, 32 worlds (11/11/10), 16 CPU threads, GPU neural work,
32-action sequences, 2 s episodes, fresh Adam .0001, seed 95023. Retain the prior
4:4:1 task weights and 10/2 ground/hover wing weights. No additional ground
non-wing distillation penalty: it would conflict with the full stand pose target.
Ground collection executes the student. Hover collection executes the complete
state-based teacher to acquire sustained stroke histories; the previous 80%
mixture still failed 192 hover training episodes. Assistance is learning data,
not evidence that the actor can fly.

Review seed 95033, 5 s per command, all student outputs, no teacher, resets or
runtime pose corrections. Retain all original gates and every failed case.
Archive actual time, experience counts, graph identity, full evaluation and a
15 s 1x video, and open it after decoding and inspection. No checkpoint is
promoted on imitation loss or assisted episode survival alone.
