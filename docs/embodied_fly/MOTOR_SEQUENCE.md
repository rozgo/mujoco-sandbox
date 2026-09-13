# Continuous commands on the canonical fly

Declared before the full-brain run on September 13, 2026. Evaluate the preserved
position_sustain_retention_01 checkpoint, with no optimization. The current
wing_position physical contract must match the checkpoint. All 78 actions pass
to the environment; no motor mask, teacher, utility selection or live reset.

Three worlds use seed 99103 and each run stand, walk, stop, resume for two
seconds per phase. Walking requests 1 cm/s forward and zero lateral/yaw motion;
stand and stop request zero. Physical state and recurrent memory persist across
every command boundary. Both pre-action and post-action poses are captured so
continuity can be checked directly, along with causal previous-action inputs.

Use the existing transition tracking, support and stability gates. Standing and
stopping also require late joint-group RMS deviation below 0.15 rad and maximum
body-height loss below 10%. All phases require wings within 0.2 rad of their
resting pose and wing speed RMS below 2 rad/s. Keep incomplete phases, tracking
failures, numerical failure details and all three video cases. Stability alone
does not establish command success. No acceptance threshold will be relaxed
after seeing this run.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_sequence \
  --checkpoint assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph /path/to/malecns --device cuda --neural-view \
  --output outputs/embodied_fly/position_sequence_01
uv run --project experiments/embodied_fly --locked python -m embodied_fly.montage \
  outputs/embodied_fly/position_sequence_01 \
  previews/embodied_fly/position_sequence_01_all_worlds_v1.mp4
```

The legacy transitions module's metrics are reused, but its legacy walking body
and 59-action mask are not used. This check tests a necessary motor
capability for later needs-driven behavior; it does not test utility, flight
transitions or the survival arena.

## First result

[position_sequence_01](runs/position_sequence_01/SUMMARY.md) is stable but fails
walk/resume in all three worlds. The opt-in training curriculum now changes
ground commands while retaining physical state and recurrent memory. Switching
worlds use current-state walking-reference labels: the frozen parent itself
does not start walking after standing, so copying its output would reinforce
the observed failure. Fixed-command rehearsal worlds and hover supervision use
the same actor. This is training exposure, not evidence that switching is learned.

## Training

[Declared first pilot](runs/position_commands_01/PLAN.md). The new options are
`--ground-switch-seconds 1 --transition-worlds 12` on the existing motor_focus
train CLI. They require the position body, all tasks, initial-form posture,
ground retention, all parameters trainable and zero ground teacher action blending.
The existing hover-mixture option may provide training-only airborne guidance;
report it separately and always evaluate the student without it.
Default zero switch interval preserves earlier recipes. At true episode
failure/timeouts, restore the original command before the normal episode reset.

Report both command-transition results and the unchanged fresh-start motor
review. Each evaluation uses one checkpoint, all 78 actuators and no teachers.
The command sequence is an externally requested task schedule; it is not a
scripted gait or learned utility selection.

[position_commands01](runs/position_commands_01/SUMMARY.md) learns start, stop
and resume in all three worlds, with moving speed 0.973-0.981 cm/s. Full gates
still fail on mean yaw. Hover also regresses, so this is a useful command-learning
result rather than the completed motor release. The recovery pilot retains
switching but also fails hover; both all-command reviews remain available.
