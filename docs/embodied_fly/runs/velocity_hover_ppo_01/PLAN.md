# Velocity hover PPO pilot

User approved the proposed ten-minute PPO pilot: return to the preserved
5.6-second checkpoint, unchanged brain/body/physics/interface, zero-velocity
commands, a recurrent replay check, hover rewards and a light imitation anchor.
First work timestamp: 2026-09-14 08:10:47 UTC. This explicitly supersedes the
earlier all-movements-per-episode requirement for this bounded hover-only stage.

Parent: `velocity_imitation_30m_01.pt`, SHA256
`c4a87e93a9fffb4679c84a6bca42716bc2e89365d48be97d9ec37780df2779e4`.
Preserve its actor and normalization, all 166,700 MaleCNS neurons and the fixed
25,582,938 signed connections. Encoder, cell gain/leak/bias and motor readouts
remain trainable. Utility heads remain disabled/frozen. One actor controls all
78 outputs. Initialize fresh PPO Adam, exploration parameters and a fresh critic;
do not carry supervised Adam moments into the new objective.

Exact accepted `wing_motion_agile_v5` body/force contract; CPU MuJoCo/mjbatch at
1,000 Hz physics, actor/observations 500 Hz, four neural updates per action.
Wing forces read measured angles/speeds every physics tick. No physical changes,
teacher actuator assistance, position targets, hidden oscillator inputs or new
sensor channels. Critic-only features add causal reward-window averages and
height (distance to the failure boundary); these never enter the actor.

32 physical worlds, 16 physics threads, RTX 4090 neural learning. Fixed starts
from original training episodes 0..7 repeated four times. No new random reset
curriculum. Rollouts 512 actions (1.024 s/world); recurrent sequences 64 actions
(128 ms, approximately four teacher wingbeats). Two PPO epochs, actor LR 3e-6,
clip 0.2, target KL 0.02, gradient norm cap 1. All 78 bounded controls use the
same tanh-Normal distribution, initial latent standard deviation 0.003, minimum
0.0005; no extra actuator noise or entropy bonus. Discount e-fold 2 s, GAE lambda
e-fold 0.25 s (combined trace about 0.222 s). Episode limit ten seconds, correct
terminal-state bootstrap for time limits and no bootstrap after physical failure.

Separate standardized 128/128 critic, LR 3e-4, four shuffled sample epochs per
rollout, including when actor KL stops further updates. First rollout fits only
the critic. Recurrent replay uses recorded sequence-start memory and reset masks;
check all first-rollout action means and log probabilities before updating.
After updates, rebuild the final sequence's carried memory with current weights;
older detached boundary memory remains a standard truncated-recurrence approximation.

New reward only: per-second rates 1 alive, up to 2 vertical velocity tracking,
1 horizontal tracking, 0.5 low rotation and 0.5 upright. Tracking uses
`1/(1+normalized_error_squared)`, velocity scale 0.5 cm/s, angular scale 0.5 rad/s.
Score a causal 100 ms window (three 30 Hz wingbeats); this does not filter physics
or observations. Multiply rates by 0.002 s. Terminate and charge a single -2 if
height falls below 5 mm, upright below 0.85, or forbidden support exceeds 0.1
body weight. Live reward rates stay positive so failure cannot avoid accumulated
negative running costs. No wing-motion penalty and no absolute-position reward.

Light imitation anchor: once per rollout, on its first actor update, add
`wing MSE + 0.1 * nonwing MSE` to the gradient. Eight sequences of 64 target steps
with 64 context steps, using only the initial two-second hover from teacher
training episodes 0..7. One sequence starts with zero neural memory; padding is
masked. No held-out examples, new recovery dataset, DAgger or noisy teacher
collection in this pilot. Count and time this learning separately within the
total training budget.

Before learning, capture PID and frozen parent on identical ten-second hover
starts 0,1,8,9. This measures the parent on uninterrupted hover; its historical
5.6 seconds came from the mixed command exercise. Predetermined midpoint and
final evaluations use the same starts, deterministic actor actions and no
physical resets after initialization. Record failures and 0.4 s continuation.
Report airborne duration, velocity RMS, accumulated displacement and rotation;
compare the first two seconds only when both runs complete that interval.
Milestone: ten seconds airborne in all four starts with at least 20% lower
matched-window velocity RMS than the parent. Report survival without low drift
as incomplete progress. No claim that other movement commands are retained.

600 seconds of physics collection plus optimization, finishing the active
rollout/update; exclude setup, recurrent audit, checkpoints, physical evaluation
and rendering. Evaluate near 300 s and at the end. Preserve final and midpoint
checkpoints; final is primary, with no retrospective checkpoint selection.
Do not extend this pilot automatically. Record and open a 1x comparison showing
the PID, parent and final actor, with common plot scales and explicit failures.

## Pre-update implementation checks

The GPU initially held an LFS pointer for the parent. Fetching that exact asset
restored the expected SHA256; no neural updates ran in that failed start.

On the first real GPU rollout, maximum action replay difference was 5.29e-7
and maximum joint log-probability difference 0.00250244 over 16,384 transitions.
The original 0.002 absolute log-probability threshold stopped before any actor
or critic update. With small action variance and 78 summed channels, tiny
float32 differences accumulate. The revised check retains the 2e-6 action
limit and requires both maximum log-probability difference below 0.01 and
aggregate numerical KL below 1e-6 (versus the learning KL threshold 0.02).
Measure and archive all three quantities; no actor/dynamics change accompanied
this numerical tolerance correction. The pre-update PID/parent physical captures
are reused after checking their physical contracts and trajectory hashes.

The full fly suite passed 223 tests in 167.71 s. The revised numerical check
also passed the focused three tests in 1.31 s. The completed run's report will
record its actual source commit, times and physical outcomes.

Subsequent user steering, 2026-09-14: "if it fails continue self improving until
we start getting results." Further evidence-driven iterations are now authorized.
Preserve this pilot and its declared settings; record each subsequent change
and measured allowance as a separate run. Physical progress, rather than offline
loss or critic fit, governs whether to continue a checkpoint or return to parent.
