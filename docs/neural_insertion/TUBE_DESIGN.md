# Thread-tube design

October 6, 2026, with the user, after reviewing photos of a current insertion
robot: prove the robot's movements (positioning, insertion, release, moving
between sites) and abstract the thread handling rather than engineer a
cartridge. This design supersedes the cannula-and-latch design
([MODERN_DESIGN.md](MODERN_DESIGN.md)).

## Design

- **Thread tube** on an arm from the tool head, beside the needle, 15° from
  vertical; it moves with the tool. Bore Ø 200 µm, 9 mm long, capped at the top.
- **Thread**: 40 µm, 8.25 mm, waiting in the tube on the bottom of the bore,
  its plain end 1 mm below the needle point and 1 mm above the tissue, on the
  needle's path.
- **Needle** on the insertion carriage: conical point, Ø 50 µm tip section,
  Ø 100 µm shaft.
- **Tissue**: the phantom with the needle-tissue force model (puncture,
  cutting, shaft friction) of `tissue.py`.

## Stand-ins (stated plainly)

All are MuJoCo `connect` constraints switched on and off by the cycle:

| Stand-in | Real counterpart (not modelled) |
| --- | --- |
| Tube hold on the thread's back end, released when the needle has the thread | feed brake |
| Needle bond, formed when the point reaches the thread's end (within 30 µm), off at depth | chemical attachment and release |
| Tissue grip: each segment entering the tissue is pinned and slips above 16.7 µN (per 0.46 mm segment, the earlier per-length value) | tissue holding the thread |

## Fast thread physics

Discrete elastic rod, 18 segments of 0.46 mm, 50 µs RK4 step (was 5 µs),
contacts settling in 200 µs (was 20 µs), solver tolerance 1e-8, and an
armature of 3e-17 kg m² per thread joint that slows the thread's fastest
wiggles (shape and sag unchanged). One cycle runs about 10 times slower than
real time on one Mac core.

## First cycle (`tube_cycle.py`, scripted, no disturbances)

Needle sticks to the thread end (0.042 s); insertion at up to about 200 mm/s
takes the end 2.04 mm deep through a puncture; release; snap back; the tool
lifts 5 mm and the thread stays 2.05 mm deep, standing out of the tissue. No
robot contact with the tissue. The thread's end springs about 0.1 mm sideways
at release: placement 83 µm from the target. Video:
`previews/neural_insertion/tube_design/tube15_full_cycle.mp4`.

How it got there, each failure seen live in the viewer:

1. The thread, built on the bore's centreline, settled 80 µm onto the bore and
   the needle missed its end: the thread is now built resting on the bore.
2. Tube at 45° with 6 segments of 1.4 mm: pulled straight down, the rigid
   segments could not bend at the tube's mouth and jammed.
3. 45° with 18 segments: at insertion speed the thread was dragged around the
   sharp corner and wrapped against the needle; tube at 15° instead.
4. An explicit tissue spring-damper on the 0.8 µg segments diverged at the
   50 µs step once the bond let go: the grip became a soft constraint.
