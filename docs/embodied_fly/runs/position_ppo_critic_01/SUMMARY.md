# Critic scheduling fixed; hover continuation regresses

The critic now completes all requested updates after policy KL stops. The
measured run confirms **208 critic updates across 26 rollouts**, instead of the
previous one update per rollout, with only **0.345939 s** spent on independent
value fitting. This corrects the training limitation but does not establish
improved physical control. The new actor loses nominal hover, so it is **not
promoted** over `position_ppo_timing_02`.

The user-authorized continuation retains the parent's actor, critic, both Adam
optimizers and exploration state. Body, graph, actor/critic architectures,
observation definition, 78 actions, reward, clocks and other hyperparameters
are unchanged. New physical episodes and the declared reset curriculum start
with seed99703. Source commit: `c4552613d8fad669f030d99a5e55970aadcac11c`.

## What changed

`--independent-critic` saves the normalized observation and preceding descending-
cell state during collection. The actor keeps its KL stop. The critic makes two
complete passes over those detached features and fixed timeout-aware GAE targets.
It needs no actor recurrent replay or extra physical experience. The deployed
brain has no added module. Historical commands without the flag keep their
original coupled schedule.

All 26 rollouts trip the actor stop after one policy update, and the critic
still completes eight updates each time. Actual optimizer step differences
verify exact continuation: actor Adam +26, critic Adam +208. Critic sample
presentations are 1,703,936, twice the 851,968 collected physical transitions;
reusing a sample does not add simulated experience.

## Physical outcome

Same nominal seed97013, five seconds per command, one frozen checkpoint, no
teacher, critic, exploration or live resets during evaluation. Initial physical
states and observations match the sustained parent exactly.

| Command | Sustained parent | Critic continuation |
| --- | --- | --- |
| Stand | Stable; passes | Stable; passes |
| Walk | Stable; yaw fails | Stable; yaw still fails |
| Hover | Airborne for 5 s | Crosses 0.5 cm height floor at 1.656 s, then falls |

Hover root RMSE worsens from 7.065 to **16.324 mm**. Full height span is
18.315 mm. Its lower full-window vertical-speed RMS includes time on the ground
and must not be interpreted as reduced airborne bobbing. Settled-hover metrics
are deliberately absent for this failed case. Walking root RMSE is 1.165 mm
(parent 2.082 mm), but yaw remains 2.673 rad/s and fails the unchanged gate.
Standing root RMSE is 0.184 mm. These are nominal-case results, not general claims.

The declared additional-start gate requires nominal hover to sustain. Therefore
no additional-start evaluation or redundant parent simulation is performed.
The previously sustained parent, including its ten-second case, remains intact.

## Critic learning evidence and limitation

Value fitting reduces overall rollout MSE in **21 of 26 rollouts**. Its first
fit reduces MSE from 0.3175 to 0.1281. Some late fits increase error: the final
rollout goes from 0.0692 to 0.1209, with hover RMSE rising from 0.3259 to 0.4651.
Hover explained variance becomes modestly positive during the run, but this
in-sample measure is neither a held-out forecast test nor a physical success.

The schedule is now exercised correctly; fitting stability remains unresolved.
The inherited critic learning rate (3e-4) was retained deliberately for this
one-change pilot. A smaller critic step is the next bounded hypothesis to test
from the preserved sustained actor before more training. The late fit increases
do not prove that critic overshoot caused the hover failure. No second run,
new reward, new physics assistance or learning-rate change is hidden here.

## Measured cost

- Setup **8.072266 s**; training **617.449570 s** (10 min 17.450 s).
  The 600-second budget finishes its last complete rollout.
- **64 worlds:** 16 stand, 16 walk, 32 hover. 16 native CPU MuJoCo/mjbatch
  physics threads, RTX 4090 neural training. No MuJoCo Warp.
- **851,968 transitions / 1,703.936 aggregate simulated seconds**;
  hover contributes 425,984 transitions / 851.968 simulated seconds.
- Collection/forward **476.675861 s**; optimization **140.762843 s**.
  Independent critic fitting **0.345939 s** is included in optimization.
- **1,379.818 transitions per training second**, including learning.
  Peak CUDA allocation **12,554,405,888 bytes**. No OOM or training exception.
- 26 policy updates, 208 critic updates, 1,703,936 critic sample presentations.
  80 stand and 80 walk training episodes complete; 60 hover episodes complete,
  229 hover episodes fail. Training episode counts include noise and reset
  perturbations and are not frozen-policy success rates.
- 5 kHz physics, 500 Hz actions, 512-action rollouts, 128-action gradients;
  gamma .999, GAE lambda .995, two epochs, actor LR1e-6, critic LR3e-4,
  initial/minimum pre-tanh noise .003, KL limit .03, five-second episodes.
- Evaluation: **7.664370 s** setup, **32.022283 s** stepping/capture,
  7,500 transitions / 15 aggregate seconds. Separate from training.

See [machine-readable statistics](MEASURED_STATS.json), [physical comparison](nominal_comparison.json),
[critic verification](critic_schedule_verification.json) and the complete progress log.

## Verification and video

149 tests pass in 128.91 s, with 43 known dependency warnings. Focused tests:
21 pass in 26.37 s. Source lint passes. Forced policy rejection leaves the actor
unchanged while the critic completes all updates. Saved-feature values, reset
boundaries, detached inputs/targets and resumed optimizers are tested.

All 229 saved training failure windows and three complete evaluation captures
are verified for hashes, finite state, bounded actions, causal observations,
selected-task rewards and one terminal penalty. The actor has 18 changed and
14 fixed parameter tensors; graph/body contracts and frozen utility/normalization
remain unchanged. Physical-reward gradients reach the modeled neuron dynamics.

[Complete video](../../../../previews/embodied_fly/position_ppo_critic_01_all_tasks_v1.mp4):
15 seconds, 750 frames, 1600x900, 50 fps, 1x, same damped camera. Rendering takes
60.573040 s. Fully decoded, 13 frames visually inspected including the fall,
and opened automatically on the Mac. No successful segment is substituted from
another checkpoint.
