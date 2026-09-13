# Fast command response; phase-dependent lift correction

All 16 physical disturbance branches change wing commands within 0–4 ms of the
first perturbed observation. Sustained scalar-lift changes cross .005 bodyweights
within 4–14 ms. The first correctly signed sustained lift change takes 4–60 ms.
Zero here means the first post-disturbance control evaluation, not zero compute
or sensor latency in a real animal. The control interval remains 2 ms.

Four of eight height probes initially have the wrong average lift direction over
20 and 50 ms (both signs at phases90/180); one of eight velocity probes does too
(rising at phase270). All 16 have correctly signed average lift differences over
200 ms. The height correction is small: around .0007–.001 bodyweights average
by that window. Immediate output sensitivity is not strong or accurate control.

No gross missing-tick or disconnected-feedback bug is exposed. This supports
one tighter vertical-speed reward continuation while retaining rates and actor.
It does not establish global bandwidth, sufficiency of every sensor, or that
stroke frequency alone causes the body ripple. Phase is an approximate measured
sweep coordinate, not a clock supplied to the actor. Scalar lift is sampled on
the last physics tick of each action and excludes passive body damping.

One nominal warmup takes9.861 s for3,047 physical actions. Twenty .3-second
branches take1.276 s for3,000 actions. Setup6.474 s, zero optimizer updates.
Each branch is initialized once from its phase snapshot; no continuing resets,
PID, injected lift or direct body servo. All sources and thresholds appear in
[evaluation.json](evaluation.json). Source b1a3db1, frozen pilot05. New reward
pilot06 will keep the actor/actor Adam and exploration, refit the critic for the
changed objective, and tighten only vertical-speed scale50 ->20 mm/s.
