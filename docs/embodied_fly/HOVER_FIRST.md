# Hover first, with a PID comparison

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
