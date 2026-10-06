# First robot motion: hold and target approach

October 6, 2026. **The robot holds without drift and reaches 1 mm above each of
the six targets with sub-micrometre settled error, no contacts and at most 17%
of any actuator's force.** This is programmed control, not a learned policy, and
no thread is handled yet. Gates were fixed in [acceptance](ACCEPTANCE.md#robot-plant)
before the first run.

## Controller

Motion comes only from `mj_step` and the scene's force-limited position
actuators at the 1 ms `implicitfast` clock. A programmed servo adds inertial,
gravity, viscous and Coulomb feedforward and a bounded integral term to each
actuator's own PD, by offsetting its position command. References are
minimum-jerk joint segments, which for this Cartesian gantry are straight
tool-tip lines.

Two design facts surfaced on the first run:

- **Z reaches only 8 mm down.** The surface is reached with the needle slide:
  Z parks 6 mm down and the insertion slide supplies the rest, leaving about
  11–12 mm of insertion travel for depth.
- **The needle and retainer hang at their retracted stops.** At zero extension
  the command range cannot offset gravity, so both are carried 0.5 mm out.
  Started from the static keyframe instead, the needle sags 0.2 mm and
  dominated the first run's tracking error.

## Results

[Manifest](APPROACH_RESULTS.json): a 2 s hold, six single approaches from the
ready pose, then one tour visiting all six targets.

| Check | Measured | Gate |
| --- | --- | --- |
| Hold, 2 s | 0 µm tip drift | < 1 µm |
| Settled hover error, single approaches | ≤ 0.19 µm lateral, ≤ 0.09 µm vertical | < 10 µm |
| Settled hover error, tour | ≤ 0.50 µm lateral, ≤ 0.09 µm vertical | < 10 µm |
| Tracking while moving, tour | 0.84 µm RMS, 4.96 µm max | reported |
| Robot–environment contacts | 0 | 0 |
| Closest needle–phantom gap | 1.158 mm | reported |
| Peak actuator force | 16.8% of limit | < 100% |

The hold's exact zero is not a frozen simulation: feedforward balances gravity
exactly and the slides' dry friction holds the remainder, while actuators carry
12–17% of their limits. Errors are against the commanded reference in an ideal
model; real encoders, backlash and thermal drift are not modelled.

## Watch it

[Six-target tour video](../../previews/neural_insertion/approach_v1/approach_tour.mp4),
real time, 1920×1080, 30 fps, 17.4 s of motion and a results card. Panels share
one simulation timestamp: overview, a following close-up, the tool-mounted
microscope and live telemetry. Camera pixels are observer output only.

![Descending to the first hover](../../previews/neural_insertion/approach_v1/frame_0063.png)

```sh
uv run --locked neural-insertion view --task tour            # live, same servo loop
uv run --locked neural-insertion approach --run-output outputs/neural_insertion/approach/recheck_v1
uv run --locked neural-insertion record --run-output outputs/neural_insertion/approach/recheck_v1 \
  --output previews/neural_insertion/approach_recheck_v1
```

`approach` and `record` exit 1 if any gate fails. The tour simulates 17.4 s in
about 1 s of wall time; recording took 36 s.

## Learning scene

A physics-only copy removes 833 zero-mass, non-colliding marking geoms and
caps the arena at 2 MB, so thousands of worlds fit in memory. It reproduces the
full scene's servo tour bit for bit and steps in 6.7 µs instead of 24.7 µs.
Vessel centerlines and the dome shape travel inside the model file as custom
numeric data for the learning environment.
