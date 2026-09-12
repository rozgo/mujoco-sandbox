# Standing failure review

[Watch the 3:02 diagnostic video](../../../previews/locomotion/standing/failure_review.mp4).
The original standing video and selected policy are preserved.

This shows one complete failed trial from each of the **15 failing CPU holdout
conditions**: healthy terrain holds, damaged flat holds, and damaged
walk–stop–walk transitions. We selected the first failing trial in each condition,
then repeated its original four-world evaluation with seed 9307 and the frozen
`consolidate_120s_seed12.pt` policy. All 15 selected result rows match the original
holdout rows exactly. This is a diagnostic selection of failures, not an estimate
of the overall success rate or every failing trial.

All 15 dogs remain upright and respect their torque caps. Some cases settle into
apparently reasonable stances but exceed the 15 cm drift limit. Others use a
prohibited support link, exceed 20° tilt, find support with a designated airborne
foot, or exceed 8 mm sampled penetration. The four penetration-only cases measure
8.32–10.32 mm. Each chapter displays its own full-trial failure reasons and limits;
none of the criteria were changed for this review.

## Watch and interpret

From the repository root on macOS:

```sh
open previews/locomotion/standing/failure_review.mp4
```

The following, overhead and detail cameras show the same recorded state. Motion
runs at 1× for the full 10-second hold or 12-second transition. Eleven chapters
then add a clearly labeled two-second paused inspection of a brief contact or
penetration peak. The total is 160 seconds of physical trials plus 22 seconds of
inspection holds. These are observer cameras, not policy inputs.

- Cyan ring: 15 cm horizontal hold region, projected onto the surface.
- Orange marker: physical limb removal.
- Red marker: the worst flagged collision link, not an exact contact-point marker.
- Contact dots: measured terminal support load.
- Left metrics: flags over the evaluated trial/window. Right metrics: current state.

Contact loads were checked every 2 ms physics step. Recorded states and penetration
samples are 20 ms apart. A contact inspection shows the state at the end of the
20 ms interval containing its peak, so the precise peak-force instant may lie
between captured states. Millimeter penetration can be hard to judge visually;
the displayed depth comes from collision geometry, not the observer markers.

| Video time | Condition | Selected trial's failure |
| --- | --- | --- |
| 0:00 | Healthy, rear-left support gap | Designated foot finds support |
| 0:10 | Healthy, rear-right support gap | Unintended support, tilt, foot finds support |
| 0:22 | Healthy, extreme pads | Drift 17.6 cm |
| 0:32 | Healthy, 18° X slope | Drift 19.7 cm |
| 0:42 | Healthy, 24° X slope | Unintended support, drift 25.5 cm |
| 0:54 | Healthy, 24° Y slope | Drift 18.4 cm |
| 1:04 | Healthy, 20 cm steps | Unintended support, tilt |
| 1:16 | Healthy, 28 cm steps | Unintended support, tilt |
| 1:28 | Lower FR removed, hold | Brief FL housing contact |
| 1:40 | Whole FL removed, hold | Brief RR housing contact |
| 1:52 | Lower FR removed, transition | Penetration 10.32 mm |
| 2:06 | Lower RL removed, transition | Penetration 8.50 mm |
| 2:20 | Whole FL removed, transition | Brief RR housing contact |
| 2:34 | Whole FR removed, transition | Penetration 8.32 mm |
| 2:48 | Whole RL removed, transition | Penetration 8.34 mm |

## Reproduce

Run from the repository root. The recorder uses CPU MuJoCo through mjbatch on
macOS; no training or GPU physics was run for this diagnostic. The existing policy
was originally extended using MuJoCo Warp and learning on the RTX 4090.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked \
  python experiments/adaptive_locomotion/scripts/record_standing_failures.py \
  --output previews/locomotion/standing/failure_review_reproduction.mp4
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked \
  python experiments/adaptive_locomotion/scripts/qa_standing_failures.py \
  previews/locomotion/standing/failure_review_reproduction.mp4
```

The recorder saves exact model binaries and trajectories in the ignored
`outputs/locomotion/standing/failure_review/` cache. Later camera/layout changes
replay those verified states with `mj_forward`; they do not advance physics again.
On a clean checkout the script recaptures from the checked-in holdout settings.
The evaluator change only permits selecting the captured world and logging each
penetration sample; default callers retain trial zero and all acceptance math.

Encoding: **1920×1080, 25 fps, 4,550 frames, H.264**. Every frame decoded
successfully. All 15 chapter samples and full-resolution opening, settled drift,
contact peak, penetration peak and ending were visually inspected. No additional
learning, reward changes, force changes, hidden resets or threshold changes.

Exact trial results, checkpoint/model/trajectory hashes, source provenance and
chapter times are in [the recording report](../../../previews/locomotion/standing/failure_review.json).
The [QA report](../../../previews/locomotion/standing/failure_review.qa.json) keeps
video integrity separate from physical task failures. Capture source hashes are
retained because capture preceded formatting and presentation-only changes;
the final render uses commit `0d8b796`.
