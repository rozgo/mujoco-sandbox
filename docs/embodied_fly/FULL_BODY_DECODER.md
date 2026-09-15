# One general decoder for the whole fly

User direction: remove the isolated wing decoder; legs, antennae, mouth and wings
must use the same decoder. Show the current learned behavior and prediction
result in a new video. Effort started September 14, 23:52:33 UTC.

The active migration combines the previous 256-unit motor layer and 128-unit
correction layer into **one 384-unit hidden layer and one dense 78-output layer**.
There is no wing selector, separate branch or output addition in live decoding.
Every output can learn from every hidden unit. The initial weight values preserve
the preceding function; zero initial connections are trainable, not masks.

The decoder consumes the same 815 MaleCNS motor-cell values. Generic preprocessing
retains two views of those values: LayerNorm and standardized/clipped LayerNorm,
yielding 1,630 input features. Keeping both preserves the prior input clipping
exactly, including outlying states. Neither view corresponds to a body part;
both enter the shared dense layer. No extra recurrence or direct sensor bypass.

This is an algebraic consolidation, **zero optimizer updates**, not new learning.
The historical checkpoint remains preserved. The active checkpoint records a
fresh-optimizer requirement because the decoder topology changed. Full-body
training can update all decoder parameters, including normalization and every
output row; cached training must use raw motor-neuron values, not stale normalized
features. The old wing-only PPO CLI remains a historical experiment, not the
recommended training path for this checkpoint. Future model-guided work must use
the full-body decoder and assess its complete motor outputs.

`fresh_velocity_actor` now initializes the shared decoder by default. Its explicit
`--legacy-wing-readout` flag reproduces old initializations only. Old policies
continue to load with their recorded architecture; they are not silently changed.

Verification: exact real-number function preservation, numerical action checks
including clipped inputs, unchanged upstream tensors, no wing-branch tensors in
the new checkpoint, gradients on all 78 rows and newly available connections,
then four full ten-second MuJoCo flights against the retained policy's captures.
The video must identify persistent drift and distinguish learned flight from
world-model prediction. Neural images show actual simulated latent activity
binned at measured cell locations; they do not claim physiological spikes.

The migration passes the complete **286-test** suite. On the real GPU graph,
2,048 motor-state samples differ by at most **0.000000637** in normalized action.
The full recurrent rollout is not bitwise identical because the dense matrix
operations use a different floating-point summation order. In matched physical
captures, maximum position discrepancy is **0.035 mm** over ten seconds; all four
starts survive. Mean velocity RMS is **12.757 mm/s** (the existing rolling-velocity
metric from 0.2–10 s, averaged across starts) and mean net climb is
**60.720 mm**, effectively preserving the retained behavior. This does not solve
stationary hover or establish leg, antenna or mouth task skills.

An initial capture used a different observation-refresh setting from the parent.
It was superseded after restoring the parent's explicit setting, `refresh before
actor`. Its results are not used to claim equivalence. The accepted capture uses
the same body, force law, starts, observations and commands as its parent.

Active checkpoint: [full_body_decoder_01.pt](../../assets/embodied_fly/diagnostics/full_body_decoder_01.pt).
[Selection manifest](PREFERRED_HOVER.json), [migration report](runs/full_body_decoder_01/migration.json),
[physical comparison](runs/full_body_decoder_01/capture.json) and
[preserved parent selection](runs/full_body_decoder_01/parent_selection.json).
The old parent checkpoint is unchanged. Migration took **1.524 s**; accepted
capture and observer predictions took **53.706 s**. Neither is training time.

[Watch the current progress video](../../previews/embodied_fly/full_body_progress_v1.mp4):
four ten-second flights, actual simulated neural activity and all 78 commands,
followed by 200 ms world-model forecasts on start 8. Forecasts are conditioned on
recorded future actuator commands; they are an observer audit, not a controller
input. The final card reports the preceding [two-minute predictor experiment](world_model/RESIDUAL_02.md),
whose improved predictions have not yet changed the actor. Playback is 1x.

Reproduce on the GPU with the existing uv-managed environment (set `FLY_GRAPH`
to the local processed graph). Raw captures stay under ignored `outputs/`:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.full_body_capture \
  --checkpoint assets/embodied_fly/diagnostics/full_body_decoder_01.pt \
  --graph "$FLY_GRAPH" \
  --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --parent-capture outputs/embodied_fly/velocity_hover_ppo_11/midpoint \
  --world-data outputs/embodied_fly/world_data_01 \
  --world-model assets/embodied_fly/world_models/residual_02/prober.pt \
  --output outputs/embodied_fly/full_body_capture_replay

uv run --project experiments/embodied_fly --locked python -m embodied_fly.full_body_video \
  --capture outputs/embodied_fly/full_body_capture_replay \
  --output previews/embodied_fly/full_body_progress_replay.mp4
```

Headless Linux rendering uses `MUJOCO_GL=egl`; do not set it on macOS.
Open the delivered video on Mac with
`open previews/embodied_fly/full_body_progress_v1.mp4`.

Next: a bounded model-guided learning trial on **all parameters of this shared
decoder**, with no wing-only parameter selection. The predictor can guide small
changes, but the real MuJoCo loop must judge them: retain all four complete flights
while reducing climb and total velocity error. The prediction result alone does
not justify promoting a changed actor.
