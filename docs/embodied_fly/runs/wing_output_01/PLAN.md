# Train only the existing wing-output parameters

Parent: state_hover_retention02, SHA-256
`f5d383485f37478b02b74cf9502c603f35b24e2f08b35e0ca6035b76b0bd87c6`.
This is a staged parameter subset of the same397-input/78-output MaleCNS actor.
No new policy, neural branch, runtime controller, action mask or physics changes.

Freeze the sensory encoder, all cell dynamics, utility/intention parameters,
motor hidden layers, and72non-wing rows of the final motor linear layer. Learn
the six existing wing rows14–19 and their biases: **1,542 eligible parameters**.
The containing PyTorch tensors have20,046parameters requiring gradients;
training-only gradient hooks keep every non-wing row at zero gradient. Adam
starts fresh without weight decay. Verify every frozen scalar remains unchanged.

Pilot settings:180-second learning allowance,32worlds(11stand/11walk/10hover),
16native CPU MuJoCo threads,5kHz physics/500Hz control,RTX4090 graph and learning,
32-action chunks,2-second episodes,seed93001,fresh Adam1e-4,gradient norm limit1.
This larger step applies only to the selected output rows; upstream gradients
are deliberately absent. Keep previous targets/task weighting:ground4:4 and
hover1; extra ground wing MSE10 and hover wing MSE2. No synthetic response loss.

The actor executes every physical action. A frozen copy of the parent supplies
ground targets; initial-pose corrections replace its six ground-wing labels.
The measured-state hover reference supplies flight labels. References never
execute a command. Compare non-wing actor/reference outputs on identical actual
sensor histories across all collected world/actions, allowing at most1e-4
normalized-action roundoff difference. Fixed parameters do not guarantee identical
physical ground motion: changed wing feedback still enters the same brain.

After training, use the same single-checkpoint unassisted5s-per-command review
at seed72001, including all startup/failure frames. Decode, inspect and open the
full video. Preserve02unless the complete result improves. Record timing, actual
transitions, selected-row gradients/changes and frozen-parameter audit explicitly.
