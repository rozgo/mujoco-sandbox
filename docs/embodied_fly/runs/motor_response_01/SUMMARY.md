# Frozen wing sensory response

Actual recorded histories of preferred motor06 are replayed through its frozen
brain. At four captured times, compare positive/negative perturbations of each
wing angle and speed. Both sensory paths are recomputed; every other input and
prior neural state is held fixed. This is not a physical rollout.

The average response is much weaker than the corrective teaching target, with
some responses pointing in the wrong direction. Exact diagonals and cross-axis
responses are preserved in report.json; aggregate ground statistics are in
summary.json. Normalization alone is not established as the cause.

- 75 parallel neural sequences; zero physical transitions and training updates.
- Setup: 2.563717 s; diagnostic: 3.985166 s.
- Replayed actions match the recorded history within 5.67e-7.
- Same checkpoint, graph, observations and physical identity.

Next train explicit local feedback using paired sensory perturbations and the
corresponding corrective labels. Keep nominal physical-state imitation, the
same body, the same deployed brain and the existing input scaling.
