# Unchanged continuation: failed, preserved

The second requested ten-minute interval continues pilot 01's actor, critic,
Adam states and exploration at the same 64 worlds and reward. It omits repeated
critic warmup. Measured training is **602.952 s**, **1,343,488 transitions**,
**41 actor updates / 328 critic updates**. All transitions are hover; no PID or
imitation enters a policy world. Source `79cf7ff`.

All three deterministic ten-second starts fail at **0.794, 0.796 and 0.804 s**.
The matching PID still passes, with an identical result to prior captures.
The actor's lower nominal position RMS includes time on the floor and is not
improved hover. Its checkpoint is preserved as a failed continuation.

The [video](../../../../previews/embodied_fly/hover_only_pid_comparison_v2.mp4)
keeps the complete capture and labels the nominal failure. It was fully decoded,
eight frames plus a full-resolution middle frame inspected, and automatically
opened. It is ten seconds, 50 fps, 1600x900, 1x, with matched fixed overviews and
tracking detail insets. The first version remains available.

The next bounded-reward pilot explicitly rolls back to pilot 01. This interval's
compute stays in total development cost, outside that candidate's ancestry.
See [plan](PLAN.md), [statistics](MEASURED_STATS.json), [evaluation](evaluation.json)
and [video verification](video_verification.json). No improvement is claimed.
