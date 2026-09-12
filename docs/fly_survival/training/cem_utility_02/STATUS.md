# Diagnostic result: water camping

Preserved checkpoint: `assets/fly_survival/diagnostic_cem_02.npz`.
Training took 320.891701 seconds after 6.295164 seconds of worker setup: six
CEM generations, 96 physical episodes, eight parallel one-fly worlds, 768
aggregate simulated seconds and 38,400 utility transitions. Native MuJoCo and
MaleCNS ran on the workstation's CPU; this was not GPU physics or CUDA training.

Four held-out 16-second trials survived 4/4 but selected Drink for 63.36 of 64
agent-seconds, consumed 2.851 water units and no food. Neural removal produced
nearly the same result (21.545 versus 21.584 mean reward). This is a useful
short-horizon shortcut, not accepted evidence of varied emergent behavior or a
benefit from the connectome. Raw paired reports are retained in
`docs/fly_survival/evaluation/diagnostic_02.json`.

The next recipe makes needs consequential within the episode using explicitly
compressed physiology, broader initial needs and twelve-second episodes. Each
learned scorer is multiplied by a stated need relevance. The outcome reward
uses the minimum of energy and hydration so water cannot compensate for an empty
energy reserve. These are engineered rules, not learned biology. No reward is
assigned for picking a named activity. The previous recipe remains reproducible
at its recorded source commit.
