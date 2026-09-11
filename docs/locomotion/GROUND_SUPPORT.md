# Minimum ground-support experiment

Request: try only the first, smallest correction before adding phase guidance.
Started 2026-09-11 00:57:18 UTC.

Diagnosis: the healthy stride reward includes a no-flight cost, but complete-loss
training gates that entire reward off on damaged bodies. Add only
`damage_flight_weight=0.5`: subtract 0.5 per 20 ms control sample when a body with
missing joints is commanded to move and no allowed foot or distal stump carries
more than 1 N. Healthy rewards, actor observations, other reward weights,
teachers, joint limits and physical servos remain unchanged. This is a soft
training penalty, with no prescribed phase or foot trajectory. It is not a hard
no-flight constraint. Force sampling differs from the healthy stride tracker's
1/5 N contact hysteresis; both use a 1 N support floor.

One 120-second continuation from `limb_bilateral_selected_seed2.pt`, seed 2,
512 environments: healthy 128, each of eight complete-removal bodies 48.
Keep the selected parent's configuration, including frozen healthy teacher
`healthy_balanced_405s_seed2.pt` and damaged teacher
`limb_bilateral_iter200_seed2.pt`; reference tensors are unchanged. CPU MuJoCo
batch stepping, MPS learner. Retain checkpoints at iterations 50 and 100 plus the
last update (if present), and preserve all outcomes.

Before tuning, fix development seed 9147 (8 trials per body), final seed 20260919
(32 per body), demonstration seed 9143. Each trial is the existing 12-second,
5 m lane task at a 0.55 m/s command. Selection requires all 72 valid development
completions, existing healthy gait gates, and each damaged body's speed at least
90% of the smaller of 0.55 m/s and its parent's speed. Among eligible candidates,
prefer the lowest equally weighted airborne fraction across the eight removals.
Accept a candidate only if average airborne fraction falls by at least 25%,
average unsupported-force fraction falls, and no body's airborne fraction rises
by more than 2 percentage points. Otherwise keep the existing policy and report
the first intervention as insufficient before proceeding to another reward idea.

Metrics use the fixed 1–12 s window, sampled every 20 ms. Unsupported force means
no allowed tip exceeds 1 N; geometric flight means all allowed terminal sphere
bottoms are more than 1 mm above the flat ground. Weakly loaded or barely separated
contacts can count in the first metric but not the second. Prohibited support is
still checked every 2 ms. No inference about unsampled airborne intervals or
natural canine gait is implied.

Results and commands will be added after this single experiment.
