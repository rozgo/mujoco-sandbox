# State-based hover learning with ground retention: declared pilot

Parent: ground_outcome_03, SHA-256
`c072fa1674c838debaecbf409e411de8e9077641e78b65cdda66b7510a6690ac`.
Preserve its body, 397 inputs, 78 actions and MaleCNS graph. No physics, force-law,
sensor, architecture or utility changes. One deployed actor for all commands.

The training stage is online corrective imitation, not PPO. A frozen copy of
the parent supplies ground body-action labels from the actual visited states,
with independent recurrent memory. Initial-pose corrective labels replace only
its six ground wing targets. The state-based hover reference supplies flight
targets. The frozen actor and reference algorithm are training-only.

- Three-minute learning allowance; 32 worlds: 11 stand, 11 walk, 10 hover.
- Native CPU MuJoCo/mjbatch, 16 threads, 5 kHz physics, 500 Hz control.
- RTX 4090 executes graph inference and gradient learning.
- Fresh Adam, learning rate 1e-5, 32-step recurrent training chunks.
- Ground worlds execute student actions only. Hover collection uses 80%
  reference action and 20% student action, logged per world and failure frame.
- Normalized task-loss weights 4:4:1 (stand:walk:hover). Ground wing MSE gets
  additional weight 10; hover wing MSE retains additional weight 2.
- Two-second episodes; seed 93001. No extra reset perturbations or synthetic
  wing-response supervision in this pilot.
- Evaluate all three commands unassisted for five seconds with seed 72001,
  preserving startup and failure frames. Render, decode, inspect and open the
  complete video. Assistance during training is never an acceptance result.

Before learning, validate the selected teacher on the Linux canonical body.
Physical checkpoint fingerprints must match exactly on the GPU host. Preserve
the existing development parent regardless of this pilot's outcome.
