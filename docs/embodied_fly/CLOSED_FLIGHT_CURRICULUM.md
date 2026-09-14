# Proposed motor curriculum: move, reverse, return, hold

User steering, September 14 UTC: paired left/right, right/left, forward/backward,
backward/forward, up/down and down/up paths, ending at the original position.
The purpose is to teach corrective flight and braking that can also improve
hover. The full shared-brain survival objective remains active; utility and later
motor stages stay deferred until physical motor control is demonstrated.

The [PID plant check](runs/round_trip_reference_01/PLAN.md) uses those six paths
and a stationary control. It is reference control, not a trained fly brain.
All targets are reached and every path returns within 0.08 mm during the final
hold. The original raw-speed gate fails even on the accepted stationary PID
because of rapid wingbeat vibration; preserve this failure and its supplementary
sustained-drift analysis. No physics or live force filtering was changed.

## First learned pilot

- Continue from the preferred PID-imitation actor, with the same measured graph,
  encoder, decoder, trainable cell dynamics and canonical wing-driven body.
  One actor/checkpoint handles every target sequence and stationary hover.
- Keep 64 worlds and 16 CPU physics threads, RTX 4090 neural computation,
  1,000 Hz physics / 500 Hz actions. Initial proposal: 32 stationary hover worlds
  and 32 split across all six ordered paths. Do not silently replace the existing
  hover-only recipe; use an explicit curriculum option and separate artifacts.
- Targets start at the original position, move to one side, reverse to the other,
  return to the origin, and hold. Jitter small distances and ramp/hold timing
  across worlds and resets, with complete return and a final hold in every episode.
  Start around the plant-checked 1.5 mm offsets. Fixed routes are development
  evaluation cases, not proof of general navigation.
- Use existing requested-position/altitude sensor inputs. Targets are commands,
  never body pose writes or direct forces. No route identifier, future waypoint,
  PID clock, teacher action or utility intention is secretly fed to the actor.
- Preserve the bounded physical hover reward initially, evaluated against the
  current requested position. Record moving targets before/after steps so actor
  observations, reward and critic bootstrap use consistent simulation times.
  Report per-waypoint reach, overshoot, final origin error, sustained drift and
  all failures. Staying at the origin cannot pass the intermediate-target gates.
- Use original critic inputs with independently shuffled time/world samples.
  The measured-return diagnostic learned the shared profile much better with
  shuffling, but failed its stricter inter-world discrimination gate. This new
  curriculum tests richer experience; it is not a claim that the critic is fixed.
- Declare the complete actor/critic/discount/optimizer recipe before starting
  the bounded pilot. Keep both failed and preferred checkpoints. Evaluate the
  same actor on stationary hover and all six round trips, with no exploration.
  Compare to the reference and earlier actor, render and automatically open
  each completed training video. Do not promote a child based on survival alone.

No learned round-trip trial has run yet. The reference video establishes physical
control authority only; successful learned correction remains to be demonstrated.
