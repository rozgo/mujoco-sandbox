# Actor-only collection: standing posture passes, combined candidate rejected

Every executed action is now the actor's output. The state-based hover teacher
and frozen ground reference supply training labels only. Same canonical body,
397inputs/78actions and one MaleCNS actor; no runtime helper or action mask.

Unassisted five-second review:

- **Standing** stays upright with permitted support and passes the complete
  posture gate. Wing maximum0.16559rad, RMS0.05614rad; maximum body-height loss
  1.366%. Command tracking still fails.
- **Walking** regresses, with first recorded envelope failure at0.378seconds.
  The complete failure remains in its metrics and video.
- **Hover** still falls; full-window root error19.29mm.

Retain run02 as the better combined development controller; preserve03 as a
diagnostic. Initial qpos,qvel and all397observations exactly match the parent's
review. This is one development case per command, not generalization evidence.
No numerical warnings occur. Training episode counts are not acceptance results.

Measured training:180.599455seconds,147,456physical transitions,144updates,
294.912aggregate simulated seconds. Setup10.091920s; collection/forward145.341525s;
optimization35.245028s.32worlds(11stand/11walk/10hover),16native MuJoCo CPU threads,
RTX4090 neural inference/learning,peak CUDA allocation9,441,672,192bytes.
Fresh Adam1e-6. All task action mixtures are exactlyzero teacher contribution.
All192saved training failure traces are verified against hashes, bounded actor
actions and actual previous-action feedback. Frozen reference and utility weights
remain unchanged; internal cell parameters receive nonzero gradients.

Evaluation takes7.726628s setup and32.252596s capture. The full15-second film
contains all three commands at1x,750frames,1600×900,50fps. It is fully decoded,
visually inspected and opened. Exact rendering time is in video_review.json.

Checkpoint SHA-256:
`980e5b2e8e5220c692b1794e408babfa005552e0a4129390c0fc8573e6ec5fa4`.
Source d641c40. No accepted motor release is claimed.

Next available test: start from02 and freeze the existing sensory encoder,
recurrent cell parameters and non-wing output rows. Train only the six wing rows
of the current final motor layer, using actual current-state targets. This is
a staged parameter subset of the same network, not an extra controller. Verify
unchanged upstream/body-row parameters and matching non-wing outputs on identical
histories, then physically evaluate every command. Physical ground behavior is
not guaranteed: wing feedback still enters the shared brain. This stage is not
implemented or claimed successful by the present result.
