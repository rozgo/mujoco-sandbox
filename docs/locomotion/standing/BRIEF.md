# Adaptive standing and walking

Goal started 2026-09-11 23:18:31 UTC. User request: train idle on varied surfaces,
including situations in which a leg should remain unsupported in the air.
The user approved implementation and clarified whether this belongs in one policy.
Deliver one shared walking/standing actor in a new checkpoint; preserve v1.

Build and preview flat, uneven platform, gentle slope and missing-corner surfaces.
Keep live robot motion entirely physical, with existing masses, joint limits,
friction and torque caps. Missing support is actual absent terrain, with a lower
visible catch floor. Initial pose placement is initialization only. Do not impose
four contacts, equal leg loads, a fixed gait phase or a frozen body pose.

Train the healthy robot first. Mix original walking with zero-command standing,
and switch walking incentives off when standing. Extend to feasible damaged-body
cases after the first standing tests pass. Use short measured GPU rounds and
preserve every attempted run. Keep the 128×128 actor/critic hidden architecture;
observable local support information can use the existing reserved context slots.
One network handles both commands; no deployed teacher or controller switching.

Predeclared development checks, separate from training rewards: remain upright
for the full trial; no unintended load-bearing link contact above 1 N; after a
two-second settling period, mean horizontal speed <=0.06 m/s, position remains
within 0.15 m of the requested hold region, tilt <=20 degrees, and torque limits
hold. A missing-corner case must sustain an existing foot without ground contact
while the other supports bear load. Check actual support at every physics step.
Use independent evaluation seeds and retain failures. Walking retention uses the
existing nine-body gait/task checks against frozen v1. Do not relax these gates
after seeing results. Numerical penetration target remains 8 mm.

Training seeds: 12, 13, 14. Development evaluation: 9301. Final evaluation: 9307
(reserved until a candidate is selected). Video seed: 9311, cases selected before
viewing candidate results. Start with one 30–60 s round; extend only with evidence
of useful learning. Report training ancestry and total new training separately.

Deliver static preview, metrics, checkpoint hashes, CPU/Warp validation, a native
Mac viewer and a video showing support surfaces plus walk–stand–walk transitions.
Cameras: overview, following/detail and head view. RGB is observer output.
Source/checkpoints/media synchronize through Git and LFS on the feature branch.

User extension (2026-09-11): retain the gentle scenes and add substantially more
aggressive pads, slopes and steps. Added training/challenge tier: 18/24 cm pad
height spreads, 12/18/24 degree slopes on both axes, 12/20/28 cm step risers.
Preview before motion. Progressively expose the shared policy to this tier and
report failures/limits under the same gates; do not represent an untested surface
as solved. The deep step cases are deliberate reach/balance challenges.
