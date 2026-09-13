# Paired wing-feedback supervision (08)

**Not selected.** Train the existing actor to change wing commands in the
restoring direction when measured angles or speeds change. The local response
probe looks better, but physical wing-pose accuracy regresses. Keep 06 as the
preferred development checkpoint.

- Source bc97107; parent 06; seed 71008; fresh Adam at 1e-5.
- 181.009871 s training, 8.098675 s setup, 127 updates.
- 130,048 physical transitions / 260.096 aggregate simulated seconds.
- 32 worlds: 11 stand, 11 walk, 10 hover; 16 native CPU MuJoCo threads.
- RTX 4090 graph/learning; 5 kHz physics / 500 Hz control.
- Collection/forward 131.165949 s; backward/optimization 49.832624 s.
- Peak CUDA allocation 13,824,762,368 bytes.
- No executed teacher assistance; ground-posture targets, wing weight 10,
  paired-response weight 1, eight sampled ground worlds per action.
- Synthetic paired observations add no physical transitions or runtime module.
  All 78 deployed actions remain outputs of the same MaleCNS actor.

The five-second development evaluation at seed 72001 keeps both ground cases
upright with permitted support. Stand/walk wing RMS is 0.1463/0.0898 rad,
versus 06's 0.1140/0.0631 rad. Standing sag worsens from 1.01% to 11.27%.
Hover falls; every full posture/tracking gate still fails. Zero numerical
warnings. Evaluation setup takes 5.487397 s and capture 32.891485 s.

All model/checkpoint/capture hashes, bounded commands and causal previous-action
feedback were checked, including 245 failure traces from 333 completed training
episodes. This is online corrective imitation, not PPO or biological validation.
The paired loss helps explain learning response but does not establish a better
controller. Do not replace the selected checkpoint or repeat unchanged training.

[Complete review](../../../../previews/embodied_fly/motor_focus_08_all_tasks_v1.mp4).
15 seconds, 750 frames, 1600×900, 50 fps, 1× playback. Render/encode/full decode
takes 60.177483 s. All frames decode; sampled frames were inspected and the film
was automatically opened on the Mac. The failed hover remains in the review.
