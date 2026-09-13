# Smaller PPO steps retain flight; accurate hover remains open

The final bounded pilot keeps all three predetermined starts airborne for ten
seconds. It still drifts nearly 6 cm, so it does not pass the reference-quality
gate. The accepted PID remains stable in a separate, physically identical world.
The [comparison video](../../../../previews/embodied_fly/hover_only_pid_comparison_v4.mp4)
shows nominal PID and PPO at 1x with matched starts, cameras and error scales.

Pilot 04 and pilot 03 start from the same pilot-01 checkpoint and actor Adam
state, use seed 120103, reset the critic and complete four critic-only warmup
rollouts. Both use the bounded physical reward, 64 hover worlds and the accepted
1 kHz plant. The only declared configuration change is actor learning rate:
**3e-6 to 3e-7**. No PID, imitation, wing phase target or action mask is added.

| Measurement | Pilot 03 | Pilot 04 |
| --- | ---: | ---: |
| Actual training time | 306.305 s | 300.545 s |
| Physical action transitions | 720,896 | 360,448 |
| Actor updates | 18 | 50 |
| Critic updates | 176 | 88 |
| Median observed maximum approximate KL | 1.33435 | 0.02749 |
| Maximum observed approximate KL | 2.51592 | 0.03492 |
| Ten-second starts without physical failure | 0/3 | 3/3 |
| Starts meeting reference-quality gate | 0/3 | 0/3 |

The smaller step lets six of seven actor-enabled rollouts complete their eight
updates. One rollout stops after two. More recurrent optimization leaves time
for fewer physical rollouts; lower transitions/second here does not mean slower
physics. This run uses 95.872 s for collection/forward inference and 204.663 s
for optimization. Setup is a separate 7.555 s. Neural work runs on an RTX 4090;
MuJoCo/mjbatch physics runs on 16 CPU threads. Peak PyTorch allocation is
12,555,753,472 bytes. No numerical training failure occurs.

Warmup is not bitwise identical despite the matching seed and recipe. Mean
reward differs by about 1.5e-7 at the first rollout; the fourth has 22 versus
23 failed/completed episodes. We retain this discrepancy. One run per learning
rate supports the update-size hypothesis but does not prove general causation.

| Nominal measurement, after first second | Pilot 01 | Pilot 04 | PID |
| --- | ---: | ---: | ---: |
| Height span | 6.055 mm | 5.569 mm | 0.131 mm |
| Position RMS error | 34.896 mm | 35.575 mm | 0.124 mm |
| Peak position error | 58.462 mm | 59.648 mm | 0.197 mm |
| Vertical-speed RMS | 59.041 mm/s | 58.423 mm/s | 13.537 mm/s |

Pilot 04 has less height variation than its airborne parent, with slightly
worse position drift. It is an optimization development candidate, not an
across-the-board physical improvement or an accepted hover policy. The PID
result dictionary is identical across all five matched captures, including
the pre-training transfer review. No harder reset curriculum was unlocked.

All four pilots are preserved. Together they cost **1,817.722 s (30 min 17.722 s)**
of training and collect **3,866,624 transitions**, or **128.887 aggregate simulated
minutes**. Pilot 04's new-stage ancestry is pilot 01 plus pilot 04: **908.464 s**.
Pilots 02/03 count toward development cost but are outside that ancestry. Every
pilot starts from an older trained motor actor; none is training from scratch.

Source `964606d`; full current suite **174 passed**, 45 dependency warnings,
150.97 s. Tests and physical acceptance are separate. The video is ten seconds,
500 frames, 50 fps, 1600x900; startup, true drift and actual wing motion remain
visible. See [verification](video_verification.json), [raw evaluation](evaluation.json),
[statistics](MEASURED_STATS.json), [paired optimization audit](optimization_comparison.json)
and [all-pilot totals](ROUND_STATS.json). Training stops for review here; straight
flight, standing, landing, walking, takeoff and utility are not trained this round.
