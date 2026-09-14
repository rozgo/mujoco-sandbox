# Exploration noise can disrupt the retained fly without changing its weights

Frozen run 05, same physical starts and 32 parallel worlds per condition:

| Exploration | Latent standard deviation | Survive 10 s | Mean airborne duration |
| --- | ---: | ---: | ---: |
| None, deployed actor output | 0 | 32/32 | 10.000 s |
| Half the PPO noise | .0015 | 30/32 | 9.421 s |
| Original PPO noise | .003 | 18/32 | 6.630 s |

Eight starts (0, 1, 2, 3, 6, 7, 8, 9), four independent noise streams per start.
The nonzero conditions share the same Gaussian draws with different amplitudes.
These are 32 rollouts of one checkpoint per condition, not 32 independently
trained policies. All actor state-dict tensors remain unchanged. No teacher,
optimizer, force changes, episode rescue or mid-flight resets.

Of the 14 original-noise failures, 10 occur before 1 second, one at 1.094 s,
and three at 5.874, 7.860 and 9.978 s. Both half-noise failures occur before
1 second. This establishes a sampling effect in these fixed starts, with much
of the damage during wingbeat startup. It does not prove that sampling explains
every failure during changing-policy training or that less noise always learns
better. The outcome supports one controlled half-noise PPO continuation.

The sampling code uses the same independent tanh-normal distribution as PPO on
all 78 outputs every 2 ms. Zero noise uses the deployed actor action directly.
Observation timing matches PPO collection, without an additional forward pass.
The force law continues reading measured wings every 1 ms. No smoothing is
inserted in either the actor or the physical model.

Velocity/climb comparisons among survivors alone can be misleading: the original
noise leaves only 18 survivors. All per-case metrics and failure times are in
[the report](report.json); failed flights never qualify as successful hover.

## Measured cost and provenance

- Source `168d5f7`; unchanged run-05 SHA256
  `3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.
- **191.765242 s total diagnostic wall time**, including **22.941998 s setup**.
  Physical collection: **50.349333 / 50.882661 / 50.780151 s**, zero/half/current.
  Remaining time is capture compression and report IO. No training time.
- CPU MuJoCo/mjbatch, 16 physics threads; CUDA full-brain inference on RTX 4090.
  1 kHz physics, 500 Hz actor; 480,000 world/action transitions across 96 flights.
- Focused sampling/replay/optimizer tests: **27 passed**, one upstream warning,
  **1.49 s**. Unit test confirms current-noise sampling equals PPO's distribution.
- Video: **43 s**, 2,150 frames, 50 fps, 1920 × 1080, 1x. Render **186.400340 s**
  on Mac; complete decode **4.296018 s**. Visually reviewed and opened
  **2026-09-14 18:47:30 UTC**.

[Watch the fixed-start noise comparison](../../../../previews/embodied_fly/hover_exploration_01_v1.mp4).
The four shown flights are the first replicate of the predefined evaluation
starts, all surviving; the result card reports survival across all 32 worlds.

```sh
open previews/embodied_fly/hover_exploration_01_v1.mp4
```
