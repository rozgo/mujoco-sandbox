# First hover-only PPO pilot: airborne, not holding position

The implemented curriculum uses 64 hover worlds, the accepted 1 kHz plant,
500 Hz actions and the full MaleCNS actor. Two zero-initialized horizontal
feedback channels were added explicitly; graph edges and old actor tensors stay
preserved at migration. PPO then updates the actor through physical rewards.
There is no teacher, PID, phase target or imitation in a learned-policy world.

The migrated actor remains airborne for two of three predeclared ten-second
starts. The trained actor remains airborne for all three. This is a gain on one
sampled perturbed start, not proof of general robustness or accurate hover.

| Nominal measurement | Before PPO | After PPO | Matched PID |
| --- | ---: | ---: | ---: |
| Whole-capture peak position error | 61.346 mm | 58.462 mm | 0.597 mm |
| Settled height span | 5.802 mm | 6.055 mm | 0.131 mm |
| Settled position RMS error | 36.761 mm | 34.896 mm | 0.124 mm |
| Settled vertical-speed RMS | 56.932 mm/s | 59.041 mm/s | 13.537 mm/s |

"Settled" means after the first second, not that motion has settled. All strict
reference-quality gates fail for the actor. The two independently captured PID
result dictionaries are identical, providing a matched physical control.

Training: **607.919 s**, excluding **8.818 s** setup; **1,441,792 transitions**,
**48.060 aggregate simulated minutes**, **40 actor updates / 352 critic updates**.
Four critic-only warmup rollouts are included. Throughput is **2,371.7 transitions/s**,
including collection and optimization. Native CPU physics uses 16 threads; neural
training uses an RTX 4090, peak PyTorch allocation **12,553,766,912 bytes**.
The actor has **2,516,402 parameters**, critic **236,033**. All 166,700 modeled
neurons advance four internal steps per collected action: **961,386,905,600**
rollout neuron-state updates, excluding training replay/recomputation. This is
an arithmetic compute statistic, not recovered biological dynamics.

The [wing diagnostic](wing_motion_diagnostic.json) measures a dominant sweep
frequency near **9.375 Hz** before and after PPO, versus **30 Hz** for PID. Mean
absolute sweep speed is approximately 31 rad/s in all three cases. PPO has not
learned the reference's temporal pattern or position correction. Tiny-amplitude
pitch/roll spectral peaks in the raw diagnostic should not be interpreted as
large wingbeats.

The subsequent reward audit finds a flaw: unbounded running penalties plus a
small terminal penalty can favor early failure. Completed five-second episodes
average -62.548 return, failed episodes -6.171 over 1.194 s. Unequal windows do
not establish causation, but motivate the explicit bounded-reward correction
after preserving this pilot and its continuation.

Source `23eecbe`. Relevant full suite: **172 passed**, 45 dependency warnings,
145.85 s. [Video](../../../../previews/embodied_fly/hover_only_pid_comparison_v1.mp4)
is ten seconds, 500 frames, 50 fps, 1600x900 and 1x; fully decoded, visually
inspected and automatically opened. One fixed camera fit and equal chart scales
serve both worlds; matching detail insets follow each physical thorax.

See [predeclared recipe](PLAN.md), [statistics](MEASURED_STATS.json),
[training](training.json), [evaluation](evaluation.json) and
[video verification](video_verification.json). No reference-quality actor is promoted.
