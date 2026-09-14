# Reordered exercises and recovery demonstrations

User approved September 14, 2026: vary exercise order while retaining every
skill in each complete exercise; include PID recovery from student-visited
states; continue the same checkpoint and evaluate physical flight. First work
on this continuation observed 07:07:07 UTC (earlier discussion/archive time is
recorded separately). Planned additional learning allowance: 1,800 seconds,
finishing the active update. Setup, data collection, validation and video are
separate measured timers. No change to body/force law, actor, Adam rate or loss.

Parent: velocity_imitation_30m_01, final checkpoint SHA256
c4a87e93a9fffb4679c84a6bca42716bc2e89365d48be97d9ec37780df2779e4.
Preserve full optimizer and RNG continuation. Same 64 sequences x 128 supervised
steps + 64 warmup slots; eight explicit episode starts; all 50 semantic stage
IDs sampled every update. Frozen graph, current v5 physical contract, 1,000 Hz
physics / 500 Hz control, four neural updates/action. No PPO, critic or reward
changes. Explicit dataset-transition flag is required; no other recipe changes.

First check ten two-second PID starts: two cold starts, six student physical
states from the nominal run at 0.25, 0.5, 1, 2, 3 and 4 seconds, and two held-out
states from episode 8 at 0.75 and 2.5 seconds. No pose or velocity alteration:
copy saved physical states only at initialization. Restore prior joint action;
align teacher oscillator phase from measured wing sweep angle and speed, with
zero PID integral. Phase remains teacher-only. Require upright >0.85, height
>5 mm, forbidden contact <0.1 bodyweights; final 0.4-second mean speed <0.5 mm/s
and yaw rate <0.12 rad/s. Failed probes remain recorded; do not train on them.

If those pass, capture ten full 71.2-second exercises from these starts, each
with a distinct seeded ordering of all 50 stages. Keep each move with its brake.
Shuffle inverse movement pairs, reversing horizontal pair order randomly;
vertical pairs climb before descending to avoid commanding a floor collision.
This varies order without removing any skill or resetting within an episode.
Require every original stage gate and physical envelope to pass.

Combine original ten episodes and new ten: 16 training (original 0-7, new 10-17)
and four validation (8,9,18,19). Expected replay episode mix is 50% original,
12.5% new cold starts and 37.5% new recovery starts. These are sampling
expectations, not extra physical worlds running inside the optimizer. Every
sample's semantic stage uses that episode's own recorded order and timing.

After 30 minutes, use the final checkpoint. Evaluate the original nominal and
held-out exercise (episodes 0,8) for a matched comparison, plus new reordered
cold and held-out recovery exercises (10,18). Preserve failures. Render/open a
1x comparison with matched teachers, actual commands and all failure intervals.
A better imitation loss alone is not a successful result. This block does not
claim full online DAgger: it branches complete teacher recovery episodes from
saved student states; later newly visited errors may require more coverage.
