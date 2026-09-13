# Hover first, with a PID comparison

Latest completed result: [pilot06](runs/hover_only_06/SUMMARY.md). A phase-matched
probe finds prompt command changes but weak, phase-dependent lift correction.
One tighter vertical-speed PPO continuation improves mean altitude error slightly
and preserves three airborne starts; settled ripple stays 2.80 mm at 9.25 Hz.
Accurate hover remains open. See [PID comparison](../../previews/embodied_fly/hover_only_pid_comparison_v6.mp4)
and [before/after](../../previews/embodied_fly/hover_only_before_after_v6.mp4).
No further training or next motor stage has started in this review round.

The accepted PID proves nominal hover is controllable in the declared flight
model. The next stage trains the same MaleCNS actor through physical PPO rewards,
starting airborne. The PID does not execute in the learner's world or generate
training labels. Utility/needs training remains deferred.

Curriculum: hover -> straight flight/turning -> stand -> landing -> walking ->
takeoff. These stages continue one actor. Hover-only learning intentionally
dedicates every world to hover; ground rehearsal returns when more tasks are
introduced. The previous all-motor checkpoint remains preserved.

## Explicit transfer

The old actor uses filtered wing forces at 5 kHz. `hover_migrate` verifies its
recorded physical identity and all model arrays, then creates a new checkpoint
for instantaneous forces at 1 kHz. Other mechanics and the 500 Hz action clock
stay unchanged. Contact integration at a larger timestep is a physical change.

Two current horizontal target-error inputs extend the sensory encoder with
256 initially zero weights. The old observation prefix, actor weights, graph
routing and learned neural dynamics carry over. Initial outputs agree for matched
inputs/state; physical behavior on the new plant must be measured separately.
This is ideal simulator feedback, not learned visual localization. The critic
and optimizers start fresh for the new input dimensions, physics and reward.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.hover_migrate \
  --resume assets/embodied_fly/diagnostics/position_ppo_timing_02.pt \
  --output outputs/embodied_fly/hover_plant_transfer_01
```

Run the following with the verified graph cache available at `GRAPH`. Linux
headless work uses `MUJOCO_GL=egl`; keep the native backend on macOS. Select
fresh output names when reproducing a run; existing artifacts are preserved.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --hover-physical --hover-only --checkpoint-activations --independent-critic \
  --preset wing_position --resume outputs/embodied_fly/hover_plant_transfer_01/actor.pt \
  --graph "$GRAPH" --output outputs/embodied_fly/hover_only_01 \
  --seconds 600 --worlds 64 --threads 16 --episode-seconds 5 \
  --horizon 512 --sequence 128 --epochs 2 --lr .000003 --critic-lr .0001 \
  --noise .003 --minimum-noise .003 --critic-warmup-rollouts 4 \
  --target-kl .03 --entropy 0 --gamma .999 --gae-lambda .995 --seed 120101
```

`--motor-all` selects the existing full-actuator motor trainer; `--hover-only`
sets all its worlds and rewards to hover. The report records zero standing and
walking transitions. No imitation or rehearsal is included in the measured time.

## Matched evaluation

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.hover_compare \
  --checkpoint outputs/embodied_fly/hover_only_01/actor.pt --graph "$GRAPH" \
  --output outputs/embodied_fly/hover_only_review_01 --seconds 10
uv run --project experiments/embodied_fly --locked python -m embodied_fly.hover_video \
  outputs/embodied_fly/hover_only_review_01 previews/embodied_fly/hover_only_pid_comparison_v1.mp4
```

The comparison runs three deterministic learned-policy starts plus one PID
reference in independent worlds of the same compiled model. Nominal PID/PPO
starts are identical. The video uses saved physical states, matched fixed
cameras, equal plot scales and 1x playback. Startup and failures remain visible.
Additional-start results are recorded even when the video shows only the nominal
pair. Reference-quality gates, failure times and full-window errors are separate.

See the [predeclared first pilot](runs/hover_only_01/PLAN.md). Training return is
a progress signal; only the independent capture establishes physical performance.

## Reward correction after the first pilots

The first pilot remained airborne on three sampled starts but did not hold
position. Its unchanged continuation fell. Auditing episode returns exposed an
incentive problem: unbounded running tracking costs could make early termination
cheaper than continuing a poor flight. The older runs remain recorded with their
original equations, videos and failures.

The opt-in `--bounded-hover-reward` uses positive, bounded physical tracking
scores and an alive term. A valid airborne state earns at least .498/s; ideal
tracking earns at most 6.1/s. A failed step earns exactly -1. Position/velocity
scales, physics and actor remain unchanged. It supplies no PID action, wing
rhythm or force target. Returning to a poor hold is still worse than accurate
hover, but ending the episode no longer avoids a stream of negative costs.

An explicit `--reset-critic` is required when changing the hover reward. This
starts fresh value fitting while retaining the actor and actor optimizer. The
new reward's numeric returns are not comparable to the old objective. See the
[correction pilot](runs/hover_only_03/PLAN.md). These are development diagnostics;
the accepted PID remains the smooth-hover reference.

## Completed comparison round

Four bounded pilots and their PID/PPO videos are preserved. The
[latest comparison](../../previews/embodied_fly/hover_only_pid_comparison_v4.mp4)
keeps PPO airborne on all three sampled starts, but nominal drift remains
59.65 mm and the height span after startup is 5.57 mm. PID measures 0.197 mm
peak position error and a 0.131 mm height span in that same time window.
Accurate learned hover is still open; the next motor stage has not started.

The final pilot lowers only the actor learning rate relative to the failed
bounded-reward pilot. Observed update divergence drops from median 1.334 to
0.0275, and 50 actor updates fit into 300.545 seconds. Its airborne parent has
slightly less positional drift, so both remain development artifacts. See the
[complete results, limitations and training costs](runs/hover_only_04/SUMMARY.md).

Open the newest comparison on macOS:

```sh
open previews/embodied_fly/hover_only_pid_comparison_v4.mp4
```

## Exploration check and next continuation

Two frozen, five-second probes compare replicated deterministic controls with
eight independent noise trajectories from the same nominal start. At the saved
.003 latent standard deviation, three sampled worlds fail; at .001, all eight
remain airborne. The latter are closer to deterministic height variation and
produce approximately the same mean lift. These are nominal development probes,
not evidence of generalization. Both retain all failed paths.

Full-graph likelihood replay without any parameter update has mean approximate
KL1.09e-5 and2.61e-6 in those probes, respectively. It is not bitwise exact, but
these values are far below the .03 update threshold. This check concerns forward
values; it does not validate long-horizon gradient quality.

`--reset-exploration` explicitly changes only the saved exploration parameter and
its Adam moments. It preserves actor weights, actor Adam history and (unless
separately requested) critic weights/Adam. Merely supplying `--noise` on an
ordinary resume intentionally retains the parent's learned distribution. The
initial effective standard deviations are now recorded in the training report.

The [next continuation](runs/hover_only_05/PLAN.md) uses .001 noise/floor and
1e-7 actor LR while keeping the rest of pilot04's physical PPO recipe. Reduced
exploration can also make it harder to escape a poor motion pattern, so success
still requires an independent physical review rather than fewer training falls.

## Fifth pilot and the next feedback check

Pilot05 completes 192 training episodes without failure after 614.579 s, but three ten-second evaluation starts still fail accurate-hover gates. The [v5 comparison](../../previews/embodied_fly/hover_only_pid_comparison_v5.mp4) is the latest completed hover video. See [results](runs/hover_only_05/SUMMARY.md). Slower wingbeats are learned motion, not a slower action clock: both controllers issue commands every 2 ms. The user approved phase-matched response diagnostics followed by one tighter vertical-motion PPO pilot. No new physical model, wing oscillator, direct body control or next motor stage is authorized by that change.
