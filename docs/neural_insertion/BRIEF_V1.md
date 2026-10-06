# Surgical insertion robot brief

October 5, 2026: build our own surgical robot simulation in MuJoCo, with
PufferLib 5.0 for subsequent task learning. Work on `main` in the primary
checkout; the stopped fly project has its own workspace.

The task is to acquire a fine thread, align with a target while avoiding vessels,
insert to a prescribed depth, release, withdraw, and repeat at several sites.
Later exercises add surface motion and recovery from missed pickups.

First deliverable: a static robot and work area for visual inspection, including
an XYZ gantry, insertion/retainer mechanism, microscope, thread cassette, supported
tissue phantom, target sites and visible vessels. Show overview, mechanism,
work-area, tool-clearance, microscope and cassette views before adding motion.

Static acceptance: model compiles; no unintended initial penetration; sampled
approach poses remain clear; cameras show the mechanism and task; dimensions,
moving masses and actuator bounds are recorded. User follow-up: bring in the
flexible thread first; test consistent internal unit rescaling while keeping source
parameters and results in SI. Document conversions, special settings, pros/cons
and failed tests. Accuracy takes priority over runtime. Then validate a stable
hold and one approach primitive, start alignment/tracking RL, and develop physical
pickup/insertion/release progressively. Stop at unresolved resolution/accuracy
limits rather than hiding them with altered physics.

Solver preference after literature review: MuJoCo first (published DER model),
Isaac Sim second, a coupled approach third, then other solvers. Keep the same
accuracy gates across candidates; use independent references when needed.

Full-task measurements will include alignment and depth error, peak tool/thread
loads, vessel clearance, retained thread placement and complete-cycle success.
Freeze numerical task thresholds after physical characterization. The proposed
learning gate is 90/100 complete missions for each of three predetermined training
seeds, with failed cases retained and a separate stress evaluation.
