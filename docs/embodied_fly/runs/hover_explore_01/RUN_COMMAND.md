# Reproduction

Run on the authorized CUDA host after checking out the source commit in training.json.
The graph path is an example sibling checkout; supply its actual local path.
The output directory must be new. Dependencies remain pinned by the isolated uv lock.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --hover-physical --hover-only --bounded-hover-reward \
  --hover-vertical-speed-scale 2 --reset-critic --checkpoint-activations \
  --independent-critic --critic-epochs 16 --critic-standardize-inputs \
  --preset wing_position \
  --resume assets/embodied_fly/diagnostics/pid_imitation_01_teacher_stage.pt \
  --graph ../fly-survival/outputs/fly_survival/malecns \
  --output outputs/embodied_fly/hover_explore_reproduction \
  --seconds 300 --worlds 64 --threads 16 --episode-seconds 10 \
  --horizon 2048 --sequence 128 --epochs 2 --lr .0000001 --critic-lr .0001 \
  --noise .003 --minimum-noise .003 --critic-warmup-rollouts 2 \
  --target-kl .03 --entropy 0 --gamma .9996000799893344 \
  --gae-lambda .9994001799640054 --seed 120601
```

The actor's physical contract supplies 1,000 Hz physics and 500 Hz controls.
No PID actions enter training. The recorded setup, training, evaluation and
rendering times have separate denominators. This is continued learning from
the preserved imitation actor, not a from-scratch five-minute fly brain.

Replay the recorded nominal comparisons on macOS:

```sh
open previews/embodied_fly/hover_explore_pid_comparison_v1.mp4
open previews/embodied_fly/hover_explore_before_after_v1.mp4
```

`reproduce_comparison.py` recomputes this trial's reported metrics from the
preserved parent and child capture directories in `outputs/embodied_fly`.
