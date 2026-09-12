# Fly survival lab

User authorization, September 12, 2026:

> Perfect. Make this the new goal. Lets make the best fly simulator we can. And show emergent behavior from simple rules and needs. Start now. Use GPU if needed.

The accepted plan specifies eight flies sharing a MuJoCo arena, a motor-driven
swatter, a modeled heat lamp, finite food and water, and shaded shelters. Each fly
has independent energy, hydration, fatigue, heat and health. One shared learned
utility policy chooses exploration, foraging, drinking, resting, shelter and
evasion. Existing FlyGym walking remains a programmed movement skill. MaleCNS
connectivity stays frozen; its measured contribution must be checked rather than
assumed. Train the scoring parameters with CEM from whole-episode outcomes.

The selected-fly inspector must show scores, actual action, needs, target, history,
eye view and a cached anatomical image with actual simulated activity overlaid.
Use our dark Ember machinery palette. Preview the static arena and UI before
movement. Interactive viewing and recorded video must share time-aligned data.

## Implementation and acceptance

1. Isolate dependencies, source, outputs and branch; preserve all accepted demos.
2. Compile and inspect a static scene and UI. Show overview, hazard and shelter detail.
3. Validate locomotion, finite state, bounded actuators, mutual contact, swatter
   impact, heat accumulation/cooling/occlusion and resource depletion before learning.
4. Audit per-agent sensing, needs and neural-state independence. Do not expose
   future hazard events or observer-wide hidden state to decision making.
5. Measure candidate world counts once, fix a training configuration, train for
   five measured minutes and extend toward ten only if observed learning supports it.
6. Compare untrained, hand-tuned and trained shared scoring policies on held-out
   scenarios. Audit removal of neural features and report retained failures.
7. Evaluate the same selected checkpoint with eight agents, record all trials and
   render a polished video with utility/neural overlays. Verify the full encoded video.

Reward comes from time-integrated internal well-being and damage avoidance, not
activity names. Intake requires a living fly at an available resource; needs and
resource quantities are bounded. Death disables control while leaving a physical
body in the arena. Numerical failure is separate from biological-model death.

## Modeling boundaries

Use FlyGym's documented millimeter/gram/second convention consistently; translate
reported physical values to SI in the specification. The lamp adds a simple
exposure/cooling/damage model; MuJoCo lighting itself does not burn bodies. Feeding
is resource intake, not a validated digestion or proboscis simulation. FlyGym's
position-driven stepping and reflex controller is not learned fly locomotion.
The neural model has measured wiring with assumed gains/signs/dynamics and an
engineered sensory encoder. Claims of emergent behavior require measured outcomes.

References: [FlyGym](https://github.com/NeLy-EPFL/flygym),
[MaleCNS](https://male-cns.janelia.org/download/),
[fly.ai](https://github.com/alextitonis/fly.ai),
[CEM](https://docs.evotorch.ai/latest/reference/evotorch/algorithms/functional/funccem/).
