# Fly dynamics prediction pilot

Goal started 2026-09-14 21:34:07 UTC; implementation started 21:34:29 UTC.
Previous goal turn established the objective but launched no experiment. PPO and
the earlier decoder-parameter probe remain paused for this prediction pilot.

Learn action-conditioned short-horizon dynamics from the accepted continuous
velocity PID and preserved student flights, including cold starts, drift and
established-flight recoveries. Validate predicted corrections in the unchanged
`wing_motion_agile_v5` MuJoCo plant. A prediction result is not a learned flight
policy. Preserve all existing policies and the frozen MaleCNS graph.

## Data and clocks

- The plant stays at 1,000 Hz, with 500 Hz bounded actuator commands. Predict
  50, 100, 150 and 200 ms ahead. Preserve every command and measured wing sample;
  temporal blocks may encode several commands, but never average away phase.
- Use full physical joint position/velocity/activation histories, without PID
  integrals, teacher oscillator clock, future observations or future commands
  selected in response to predicted states. Recorded future action sequences
  are supplied explicitly for action-conditioned prediction, not policy inputs.
- Keep source trajectories and all interventions derived from an episode in
  the same split: training episodes 0/2/3/4/5, validation 1/6/7, test 8/9.
  These are held out from world-model fitting; some were previously inspected
  during policy development and are not a new blind robotics benchmark.
- Exclude post-failure states. Record source report/capture hashes, physical
  identity, action alignment and reset/replay agreement.

## Model and comparisons

Use a compact history encoder, action-window encoder and recurrent latent
predictor. Train future-embedding consistency with anti-collapse regularization;
then fit a physical readout on frozen latent dynamics. This is a JEPA-style
adaptation for the fly, not an upstream checkpoint or a paper reproduction.
Use the existing PyTorch/uv stack and GPU for learning.

Compare physical displacement, velocity and rotation predictions with constant
velocity and an explicit reduced analytical wing/body predictor. The latter
has known force-law parameters but cannot read future simulated states.
Keep all comparator settings fixed across splits; choose checkpoints on
validation only. Report wall time separately for setup, data, training and tests.

## Intervention test and decision

From matched physical states, execute unchanged and small positive/negative
wing-command changes for up to 200 ms. Include different wing phases, initial
support and warmed-up motion. Keep non-wing command sequences identical. These
are offline plant experiments, not a new controller attached to MaleCNS.
Report duplicate-control noise, baseline replay error, clipping, any physical
failure, and predicted versus actual change in support and body velocity.

Before using the model to guide learning, require all of:

1. Causal data alignment, physical fingerprint and intervention replay checks pass.
2. Predictions beat constant velocity on held-out velocity error at every stated
   horizon, and the overall error is no worse than the analytical comparator.
3. For intervention effects above 0.2 mm/s, at least 80% sign agreement and
   normalized effect RMSE below 0.5, reported separately for each velocity axis
   and for cold/warm starts. Also report raw magnitudes and low-signal cases.
4. Physical support effects are evaluated as well as velocity. Any surviving
   comparison must retain failures and avoid using post-failure samples.

These are pilot selection gates, not guarantees of successful policy training.
If the model fails, document which requirement failed and whether collecting
more informative transitions, changing the predictor or using the analytical
model is supported by evidence. Do not launch extended PPO on a failed model.

Research basis: [SkyJEPA](https://arxiv.org/html/2606.23444) for latent dynamics
and physical probing; [Dreamer](https://arxiv.org/abs/1912.01603) for learning
behavior through predicted futures; [MBPO](https://arxiv.org/abs/1906.08253) for
short model rollouts anchored in observed experience. World-model guidance is
training machinery; the deployed sensor → encoder → MaleCNS → decoder → wings
path remains unchanged.
