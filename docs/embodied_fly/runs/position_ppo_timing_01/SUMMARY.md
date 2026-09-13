# Longer PPO timing: first measured result

The actor retains standing and walking stability but loses the nominal hover.
Standing passes the existing gate; walking yaw still fails. Hover crosses the
recorded 0.5 cm height floor at 2.216 seconds and falls. Full five-second root
RMSE is 14.243 mm versus 6.004 mm for the preserved parent. This checkpoint is
not promoted. Its lower full-window vertical speed includes time on the ground
and must not be described as reduced airborne bobbing.

The user authorized an additional ten minutes with more resources during this
run. [Timing02](../position_ppo_timing_02/PLAN.md) resumes this actor, critic and
optimizers with 64 worlds. Additional-start evaluation will assess that
continuation against the preserved parent. This first checkpoint and its
complete three-command capture remain independently available.

## Measured training

- Setup: 9.133460 s.
- Training: **606.981945 s**, including two critic-only warmup
  rollouts; 573,440 world/action transitions.
- 32 worlds: 8 stand, 8 walk, 16 hover. Aggregate experience: 286.72 s stand,
  286.72 s walk, **573.44 s hover**; 1,146.88 s total.
- Collection/forward: 467.722505 s. Optimization:
  139.240045 s. These timings do not isolate pure physics
  from neural inference inside collection.
- 944.740 transitions/training-second.
- 35 rollouts, 33 actor updates, 49 critic updates. KL early stopping limits
  updates; the critic still has negative hover explained variance in the last
  rollout. More samples have not established physical improvement.
- Peak CUDA allocation: 6,539,169,280 bytes. No GPU OOM.
- Native CPU MuJoCo/mjbatch physics at 5 kHz; RTX 4090 neural work, 500 Hz actor.
  512-action rollouts and 128-action recurrent gradients. No MuJoCo Warp.
- No teacher actions or imitation loss; all 78 actuator channels learned.
- 395 completed training episodes: 56 stand and 56 walk complete, 37 hover
  complete and 246 hover failures. Training completion is not evaluation success.

## Verification and video

143 tests pass, 43 known dependency warnings, 121.56 s; focused 12 checks also
pass. Recomputed recurrent outputs and gradients match ordinary execution.
All 246 saved failure windows pass checks for finite state, bounded actions,
causal feedback, reward decomposition, task selection and one terminal penalty.
All three evaluation captures match physical/checkpoint hashes and use no teacher.
18 parameter tensors changed; 14 frozen tensors and measured graph/body
contracts match the parent.

Nominal evaluation: setup 8.174788 s; physical stepping/capture
31.626228 s. Three worlds, five seconds each, 7,500
transitions. Initial physical states and observations match the original review.

[Complete video](../../../../previews/embodied_fly/position_ppo_timing_01_all_tasks_v1.mp4):
15 s, 750 frames, 1600x900, 50 fps, 1x playback. Render/encode
61.397560 s. Fully decoded; 13 frames inspected,
including the fall onset; opened on the Mac. Earlier footage is preserved.
