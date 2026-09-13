# Retain learned walking while improving sustained flight

Position_sustain01 finishes 180.682384 s of training. Its full student review
keeps standing and flight upright, with hover root RMSE 20.34 mm, but the walking
request now produces almost no motion. The preserved position_motion01 advances
about 4.9 cm in its five-second review. Neither passes every motor gate.

Return to position_motion01 as the parent. Continue for 180 seconds with the
same 32 worlds, 16 CPU threads, sequence 32, five-second episodes, learning rate
0.0001, 4:4:1 task weights and ordinary startup weight 1. All motor/cell-dynamics
parameters train; measured connectivity, utility, input format and physics stay
fixed. Ground wing weight remains 10; hover wing weight remains 2.

Use the existing --retain-ground option to copy the parent into a frozen,
training-only reference with independent recurrent state. Its learned walking
commands supply ground targets on the student's actual observations. Initial-form
standing targets and resting-wing targets still apply. Add the existing non-wing
ground MSE with weight 4 to discourage forgetting the working ground behavior.
Execute 100% student commands in every world, including walking. This changes
the training distribution as well as supervision, so it is a developmental
experiment, not an isolated causal comparison with sustain01.

There is one deployed checkpoint; the frozen parent is not loaded for evaluation.
The handcrafted measured-state reference supplies hover training labels only.
The walking reference flag is receding because --retain-ground replaces those
labels; no anchored future trajectory is used by this run's deployed actor.

Training seed 97003 and evaluation seed 97013 match sustain01 to retain the same
initialization families. Full five-second unassisted review per command, all gates
unchanged, every failure preserved, full 15-second video decoded/inspected/opened.
No claim of learned utility, takeoff, landing or the complete survival simulator.
