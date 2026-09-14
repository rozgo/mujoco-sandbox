# Closed-flight PPO pilot 01 — predeclared

Follow-up implementation started 2026-09-14 01:39:52 UTC. User proposes ordered
opposite movements ending at the original position. The accepted PID completes
these paths. This trial tests whether varied position commands help the learned
actor discover correction, braking and holding. No success is assumed.

## Fixed recipe

Preferred parent: `pid_imitation_01_teacher_stage.pt`, SHA256
`315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3`.
Preserve this parent and every previous result. One unchanged MaleCNS motor actor,
399 observations / 78 actions; same frozen graph and trainable encoder, motor
decoder and cell dynamics. Utility remains frozen. Same physical contract:
1,000 Hz instantaneous wing-driven physics, 500 Hz actor, wing_position.

- 64 worlds; 16 CPU physics threads; RTX 4090 neural computation. 32 stationary
  hover worlds, 32 split across six ordered routes, cycling route on episode reset.
- 12-second episodes. Origin -> first target -> opposite target -> origin -> hold.
  Amplitude uniformly 1.2–1.8 mm; entire target schedule time scale 0.85–1.15.
  Nominal return at 8 s, latest return at 9.2 s, final measurement at 11–12 s.
  Exact original reset, cold wings; no automatic widening in this pilot.
- Bounded physical reward unchanged from hover_explore_01: vertical speed scale
  2 cm/s, height/horizontal position scores evaluated against current target.
  Zero velocity remains the settling preference. No teacher, imitation or
  hardcoded wing timing. Targets never write body state or forces.
- Horizon 2,048; sequence 128; two actor epochs. Actor LR 1e-7; exploration
  standard deviation/floor 0.003; KL guard 0.03; entropy weight zero.
- Original critic input scaling, shuffled individual time/world transitions;
  16 critic epochs; LR 1e-4; two critic-only warmup rollouts. Fresh critic and
  actor/critic Adam state from the imitation parent. No input-standardization
  variant: its preceding diagnostic was worse.
- Gamma 0.9996000799893344; GAE lambda 0.9994001799640054; seed 120901.
- Requested training allowance 300 seconds. The loop completes its current
  rollout/update cycle, so record actual elapsed training separately from setup,
  evaluation, rendering and development. No automatic allowance extension here.

Actor sees only current requested-position error through its existing sensory
inputs. Route identifiers, future targets, timing and evaluation drift history
are observer-only. Post-step command advances before reward and critic timeout
bootstrap. Failure traces retain pre-action observation/target and post-action
physical state/target, explicitly named.

## Evaluation and decision

Freeze the parent and child separately on the same seven nominal 12-second cases:
six ordered ±1.5 mm paths and stationary hover. Exact canonical starts; no
exploration, live resets or teacher. Retain every failure through the full capture.
These are fixed development cases, not a generalization benchmark.

Success requires no physical failure, peak position error <0.5 mm in each of
three windows (3.5–4 s, 6.5–7 s, 11–12 s), and final-hold maximum 100 ms
position-displacement speed <1.5 mm/s. Merely remaining at origin fails the
intermediate gates. Report raw velocity separately; the existing PID reference's
failed raw-speed gate is preserved, not relabeled. Sustained drift is measurement
only and never filters live physics.

Do not promote on survival alone. Progress requires at least 10% reduction in
mean whole-path position RMS over the six routes and no more than 5% regression
in stationary hover position RMS, with no new physical failure. Full acceptance
still requires all seven success gates. A failed pilot is retained and explained.
Render all seven learned cases with clearly labeled PID reference comparison,
inspect/decode the video, automatically open it, archive evidence and synchronize.
