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
and 59-action mask are not used. This check establishes a necessary motor
capability for later needs-driven behavior; it does not test utility, flight
transitions or the survival arena.

## First result

[position_sequence_01](runs/position_sequence_01/SUMMARY.md) is stable but fails
walk/resume in all three worlds. A future curriculum should include changing
ground commands while retaining physical state and recurrent memory. On those
worlds, use current-state walking-reference labels: the frozen parent itself
does not start walking after standing, so copying its output would reinforce
the observed failure. Keep separate fixed-command rehearsal worlds and hover
supervision in the same actor. This curriculum is proposed, not implemented by
the evaluation module above.
