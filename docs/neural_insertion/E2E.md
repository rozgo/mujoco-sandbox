# End-to-end cycle, first pass

October 6, 2026. User direction: validate the whole cycle (pick a thread,
align, insert to depth, release, withdraw) as soon as possible, before refining
any single phase, so each stage's real problems surface early.

## Scope of the first pass

- One cycle, one site, scripted yardstick control (minimum-jerk joint references
  and the programmed servo), no disturbances. Learning comes after.
- The full DER thread (40 µm, 44 mm, 32 segments) at the task-regime clock:
  RK4, 5 µs, 20 µs contact time constant, in millimetre-gram engine units.
- A separate scene variant ([`e2e_scene.py`](../../src/sixlegs/neural_insertion/e2e_scene.py));
  the alignment scene and its trained policies are unchanged.

## Mechanism (ours, provisional)

- **Eyelet:** a rigid ring at the thread's free end, 180 µm radius, 20 µm wire.
- **Slotted needle:** two lips near the tip protrude toward -Y. The needle passes
  down through the eyelet beside the rim, moves -Y so the rim enters the slot,
  and then carries the rim up (pick, lift) or down (insert). Moving +Y slides the
  slot off the rim (release).
- **Keeper:** the retainer slide now rides on the needle carriage and lowers two
  20 µm prongs beside the slot opening during transport. It retracts before
  insertion.
- **Cassette bed:** the thread lies flat with a 0.2 mm wide, 0.6 mm deep trench
  under the eyelet so the lower lip can pass below the rim.

## Tissue (provisional force model)

MuJoCo has no puncture model. The needle tip, slot and the eyelet end of the
thread do not collide with the phantom; [`tissue.py`](../../src/sixlegs/neural_insertion/tissue.py)
applies documented forces instead: a dimpling spring until puncture (2 mN at
0.4 mm), then cutting and shaft friction, a lateral spring toward the entry
point, and soft stick-slip anchoring of thread bodies below the surface. All
magnitudes are illustrative until characterized. Every other robot or thread
contact with the phantom remains a real contact, and robot contact with the
phantom is recorded as prohibited.

## Tissue channels (the user's proposal, adopted for insertion)

Rather than invent retention and lateral constraint, the tube variant
(`--tubes`) gives the target a pre-formed channel with real collision: sixteen
wall plates around a 0.28 mm bore, 3 mm deep with a floor, and a surface skin
around the opening so a needle that misses the channel hits tissue. The bore fits
the needle, slot lips and eyelet, plus the 0.11 mm release slide, with 50–80 µm
clearance. Guidance and retention then come only from contact, friction and
gravity; the only applied tissue force left is the needle's axial cutting and
shaft friction. A needle tip below the surface outside the channel is recorded
as a miss. Limits: no puncture or dimpling, fixed insertion sites, and contact
softness and friction that are as uncalibrated as the force model's numbers.

## Checks (measured state)

| After | Check |
| --- | --- |
| Seat | eyelet lifted more than 0.2 mm off the bed by the slot |
| Lift | eyelet within 0.6 mm of the needle tip |
| Lower | eyelet with the needle and within 0.1 mm laterally of the target |
| Insert | eyelet more than 1 mm below the surface, puncture recorded |
| Withdraw | eyelet still more than 1 mm deep and more than 2 mm from the needle |

## Cost

Scene compile 0.08 s; 3.0 ms per 5 µs step on the Mac CPU, about 600 s of
compute per simulated second. A 4 s cycle is roughly 40 minutes.

## Findings so far (scripted yardstick, no disturbances)

| Run | Change | Result |
| --- | --- | --- |
| v0 | first pass | pick passes (eyelet lifted 0.33 mm); straight 44 mm plunge onto the trailing thread buckles it (joint rates near 30,000 rad/s), the eyelet is torn off, and MuJoCo resets after a bad acceleration; the runner now stops on that warning |
| v1 | carry at full Z height, diagonal laying descent | thread lost 0.5 s into the descent, as its free end lands |
| — | keeper commanded in metres in millimetre units | the keeper never closed in v0 or v1: the eyelet hung on the lower lip by gravity |
| v2 | keeper fixed, clamp pads, 3 µm squeeze | the squeeze never engaged (servo error larger than 3 µm); the rim slid under the pads and off the lip during the lift |
| v3a | fences added, 30 µm squeeze | the pads struck the eyelet and ejected it; numerical blow-up at 0.401 s |
| v3b | cage: pads at contact, gentle close | no ejection, but the ring rotates to hang vertically from the lip during the lift |
| v4 | force-limited clamp: weight cancelled, 5 mN on the rim, closing speed capped at 0.5 mm/s | clamp engaged as intended, but the few-microgram ring tilted under 5 mN and the simulation blew up 63 ms into the lift |
| v5 | 100 µN clamp (the validated press-fixture load), flexure-guided keeper | clamp engaged; thread lost 82 ms into the lift |
| v6 | latch: one bar 10 µm past the lip ends closes the slot into an eye; no clamping | pick, latch and the 44 mm lift pass with the eyelet 0.13 mm from the tip; full cycle running |

The central problem was holding a few-microgram eyelet on one lip while 44 mm
of thread hangs from one side of it. Forcing it (clamps) is numerically stiff
and fragile at this scale; caging it from above lets it pivot off. Closing the
hook into an eye, as a knitting machine's latch needle does, lets it swing but
not leave. Every change above is mechanism design or motion; the physics models
of contact, friction and the thread are unchanged.

Review videos (local, `outputs/neural_insertion/e2e/videos/`): the straight
plunge, the squeeze ejection and the cage show the three distinct lessons; the
latch video shows the hold through the lift. Viewer for live or recorded runs:
`python -m sixlegs.neural_insertion.e2e_view {live,replay}`.
