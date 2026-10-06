# Surgical insertion robot

The first static workcell is ready for inspection: five actuated slides, a
mounted microscope, a tissue phantom with six targets and vessel markings, and
a four-thread presentation cassette. [User brief](BRIEF.md).

A separate **40 µm flexible-thread benchmark** now compiles using consistent
internal unit rescaling. Bending passes preliminary analytical checks; contact
rebound has not passed convergence. See the [rescaling study](RESCALING.md) for
measured results, exact commands, conversions and tradeoffs. Simulation quality
takes priority over throughput; robot integration and RL wait at this physics gate.
The [solver decision](SOLVER_DECISION.md) prioritizes published DER inside MuJoCo,
then an Isaac/Newton route, then coupled solvers, then standalone alternatives.
It includes source compatibility findings and a proposed integration diagram.
The [DER port and measured gates](DER_VALIDATION.md) now establish consistent
forces, reproducible state handling, and preliminary bending/free-relaxation
checks. Contact refinement still fails; robot integration and RL remain gated.
The [contact-isolation study](CONTACT_ISOLATION.md) explains why rigid segments
are diagnostic controls for our flexible thread, and separately tests timestep,
mass-unit and length-unit sensitivity. Rescaling is not assumed harmless.
The [focused contact audit](CONTACT_AUDIT.md) now reproduces rigid rebound errors
with an independent fixed-step calculation and identifies contact-entry sensitivity
in the chain. Same-state force calculations agree across units. Contact-event
localization/adaptive substepping is the next proposed experiment; accuracy remains
unaccepted and no engine defect or successful patch has been established.
The robot now moves: it holds without drift and reaches all six targets with
sub-micrometre settled error ([first motion](MOTION.md)). The alignment
learning task is solved: a PufferLib 5.0 policy aligns the needle on all 200
held-out episodes, faster than the scripted reference ([learning](LEARNING.md)).
No thread is carried and the phantom cannot yet be entered.
Gates now follow the task phases ([acceptance](ACCEPTANCE.md)), and every
task-regime thread fixture passes at a 5 µs step ([task regime](TASK_REGIME.md)).
[How we got here](PROCESS.md) records the reasoning and corrections.
The [contact-smoothing experiment](CONTACT_SMOOTHING.md) then changed only two
numerical settings, a penetration-ramped contact impedance and RK4 stepping: the
rigid gates pass by 10–100×, chain unit invariance improves about 1000×, and a
low-velocity settling fixture passes including settled penetration. The 6 mm
frictional chain drop still fails the pointwise 1 µm gate because it amplifies
roundoff-level differences; that limit is now measured rather than unexplained.

```sh
uv run --locked neural-insertion view --static
uv run --locked neural-insertion view --task tour
uv run --locked neural-insertion preview
uv run --locked neural-insertion inspect
```

Viewer keys: **1–6** choose overview, mechanism, field, tool, microscope and cassette;
**F** enables the free camera. `--seconds 5` runs a bounded viewer check. This
milestone stays at simulation time zero. Previews are in
[`previews/neural_insertion/static_v1/`](../../previews/neural_insertion/static_v1/).

## Layout and parameters

SI units; +X right, +Y toward the rear of the machine, +Z up. Deck dimensions are
658 × 448 mm; the beam reaches 502 mm above the origin. The 92 × 72 mm phantom
has a shallow dome, with its center at `(0, -0.055, 0.112)` m. The needle is
150 µm in diameter; the presentation samples are 40 µm. These are the first
prototype's design parameters. Each slide uses an explicit diagonal inertia
in its body frame; the values below approximate its component envelope.

| Slide | Travel (mm) | Body mass (kg) | Inertia diagonal (kg m²) | Damping (N s/m) | kp / kv | Force limit (N) |
| --- | --- | --- | --- | --- | --- | --- |
| X | −175…175 | 2.4 | .006, .008, .008 | 12 | 12000 / 160 | ±100 |
| Y | −75…75 | 1.3 | .004, .0018, .003 | 10 | 10000 / 120 | ±80 |
| Z | −8…50 | .65 | .00065, .00065, .00045 | 6 | 8000 / 85 | ±40 |
| Insertion, downward | 0…18 | .025 | 2e−6, 2e−6, 3e−7 | .15 | 1200 / 4 | ±2 |
| Retainer, downward | 0…6 | .004 | 3e−7, 3e−7, 1e−8 | .03 | 300 / .7 | ±.3 |

Gains use N/m and N s/m; commands are metres, with unit transmissions. Dry slide
friction is 0.2 N for XYZ and 0.005 N for the fine slides. Total moving mass is
4.379 kg; the Z stage carries 6.661 N of gravity against a 40 N actuator bound.
The provisional clock is 1 ms with `implicitfast`. The next step is a hold test.
Static housing/support masses use primitive volumes at MuJoCo's default
1000 kg/m³; those bodies are mechanically fixed to the base.

The needle, tool, cassette, phantom and robot housings have collision geometry.
Default adjacent-body filtering covers the connected slide guides. Vessel/target
markings and cassette thread paths are visual layout elements; flexible thread
mechanics are a subsequent component. The phantom is currently a solid collision
mesh. Contact friction is `0.5 0.005 0.0001`, with `solref="0.004 1"` and
`solimp="0.95 0.99 0.0001"`. Shadow maps are disabled for readable microgeometry.
Cameras are inspection output.

## Validation and next work

Static checks pass: no initial contacts; no penetrations at 120 sampled approach
poses; no MuJoCo warnings; command and force bounds enabled. Needle/surface gap
is 12.925 mm initially and at least 4.925 mm across those sampled configurations.
Both focused tests pass, including collision when the needle is deliberately
extended into the solid phantom in a scratch configuration. The samples do not
cover every pose or establish dynamic stability.

1. Carry the smoothed contact law and RK4 into the DER benchmark with measured
   contact/material parameters, retaining the native-cable, published-force and
   drop-sensitivity records. Decide the thread/robot clock coupling. Then validate
   bounded hold and one target-relative approach with measured tracking error.
2. Integrate PufferLib 5.0 immediately after those basic plant checks; start
   alignment and tracking RL while physical thread handling is developed.
3. Add pickup/release, insertion resistance and retention, with matching physical
   acceptance tests and progressively richer learning tasks.
4. Learn continuous multisite placement with surface motion and recovery.

Training will use a native MuJoCo C/C++ environment adapter and bounded continuous
commands, with a single recurrent policy. Start from inspected PufferLib revision
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`; its training backend requires NVIDIA.
Keep training dependencies in `experiments/neural_insertion/` when integration
begins. Local scene/viewer use CPU MuJoCo. [Puffer documentation](https://puffer.ai/docs.html).

Implementation: `src/sixlegs/neural_insertion/`; generated MJCF:
`build/neural_insertion/`; future run data: `outputs/neural_insertion/`.
