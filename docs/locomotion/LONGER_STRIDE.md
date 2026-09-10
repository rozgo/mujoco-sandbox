# Longer-stride walking experiment

Started September 10, 2026 at **16:16:47 UTC**.

User request: “could we reward a longer gait, so it does look like its taking very short steps... its looking good otherwise”. The intended change is longer steps at the same requested speed. Preserve the current foot-valid policy and its videos.

The inspected existing gait advances each foot approximately 21 cm between placements, with approximately three cycles per second. Add an optional contact-event reward for forward swing displacement on landing, retaining the foot/stump support rule. The reward has no prescribed footfall order, phase clock, mirrored actions or target foot trajectories. It saturates at 40 cm, subtracts a 30 cm per-landing offset, and penalizes supporting foot slip and all-feet-off-ground flight. Air time alone earns no reward. The new `--stride-weight` defaults to zero so existing training behavior remains available.

Start with 90 seconds of refinement from the 208.573-second healthy policy, keeping total lineage under five minutes. Select on development seed 9137. Before final evaluation, require at least 25% greater mean touchdown-to-touchdown stride, no more than 10% increase in actual forward speed, unchanged support-valid completion, and no excessive hopping or penetration. Reserve final seed 20260913 for 64 trials and 16 half-timestep trials. Record failures as well as successes. Longer training is justified only by measured progress under the user's standing short-training allowance.
