# Shared standing and walking policy

The actor is still 86 → 128 ELU → 128 ELU → 12 joint targets; the separate
critic is 90 → 128 ELU → 128 ELU → 1 value estimate. One actor handles commands.
A frozen v1 actor supplies training-only walking targets. It is absent at runtime.

The original 66 observations retain joint positions/velocities, body angular
velocity and gravity direction, previous actions, desired planar velocity/yaw
rate, joint availability, and nine range rays. The former 20 constant actor
context inputs now carry standing-only feedback: four terminal-contact bits,
four downward ranges, body velocity and planar hold-position error (13 channels;
seven remain reserved). Those additional channels are zero while walking. Before
training, their constant contribution is folded into the first-layer bias and
their weights are zeroed, preserving the old actor output to float precision.
No new IMU hardware/noise model is added. Body-state and odometry estimates are
ideal simulator measurements. Camera images are observer output, not perception.
The exact terrain map is used for initialization and independent diagnostics,
never as an actor input. Real range rays may be occluded by moving robot links.

Every 20 ms the program computes rewards separately for each world. Standing
replaces walking's progress/gait rewards with low horizontal/angular velocity,
small hold-region drift, upright posture, a body-clearance band, moderate effort,
action smoothness, low loaded-foot slip and allowed support. It does not demand
four contacts, equal loads, a phase template or a particular foot in the air.
The hold anchor is recorded when the motion command becomes zero; corrective
steps and a small weight shift are permitted. Motor torque caps remain 23.7 Nm
for hip/thigh and 45.43 Nm for the calf. Each control target drives ten 2 ms
MuJoCo physics steps. The actor has no online learning during evaluation.

Training uses 4,096 simultaneous worlds, PPO horizon 24, four epochs, minibatches
of 3,072, Adam at 0.0001, gamma 0.99 and GAE lambda 0.95. Physics and learning
run on the RTX 4090; observation/reward assembly remains NumPy on the host.
The first three rounds split worlds between healthy flat commands and seven
gentle nonflat surfaces. Later rounds use 27 groups: half the worlds split among
all nine body topologies on flat ground; the other half split among 18 healthy
nonflat surfaces. Flat episodes are 20% stand, 45% walk and 35% walk–stop–walk.
Nonflat episodes stand. This does not train damaged bodies on every terrain.
Mass and motor-strength randomization are disabled for this first extension.

Surfaces are physical boxes with friction 0.8 and the existing firm contacts.
The catch plane is 40 cm below the central datum. A gap is an omitted platform,
not disabled robot collision. Reset-only inverse kinematics places feet near
available supports, and initializes the gap-side foot above the missing surface.
This makes initial stance feasible; maintaining balance is the learned task.
There is no live inverse kinematics, pose correction or hidden support.

The native Mac viewer runs the same policy/environment loop using CPU MuJoCo.
Acceptance measures support at every physics step. Penetration is additionally
measured by independent CPU collision queries at 50 Hz; this is explicitly a
sampled diagnostic, not proof of a substep penetration bound. Failed trials are
retained for the full evaluation window and never reset to look successful.

Predeclared video cases are in `standing_record.VIDEO_GROUPS`: healthy
walk–stand–walk with follow, overview and head views; six gentle support scenes;
six aggressive scenes; and four damaged bodies standing on flat ground.
Playback is 1× and each group is synchronized by simulation time. All cases,
including failures, are recorded from physical rollouts before rendering.

After the first three minutes, healthy standing passed multiple missing-support
cases and walk–stand–walk. The two-minute checkpoint retained healthy walking but
failed several damaged-gait retention gates. Subsequent mixed-body training adds
a fixed rehearsal dataset of valid v1 walking observations/actions from all nine
bodies (training seed 12). PPO minibatches also minimize action error on 512
random rehearsal examples. This is training-only behavior retention; there is
still one actor and no teacher, replay buffer or policy switch during execution.

Contact-window refinement: diagnosis found brief unintended support during the
first second that endpoint-only training could miss. Optional `--substep-support`
accumulates native contact force peaks and impulses at every 2 ms physics step
on the GPU, inside the captured graph, then transfers the summaries once per
20 ms action. The standing penalty uses these peaks. Physics and actuator targets
are unchanged. CPU uses explicit substeps for the same mode. Validation retains
its independent explicit-substep path. The default legacy reward path is unchanged.

The selected run uses standing support weight 8, drift weight 3 and walking
rehearsal weight 15. One short acquisition round used two-second standing
episodes; the final two-minute round restores ten-second episodes so that the
policy must sustain its hold. Walking episodes remain ten seconds throughout.
The fixed walking teacher weight is 3; all added standing inputs are disabled
while moving. The networks still share parameters across both behaviors.

All 19 scene presets are retained, including challenge failures. Slopes are
6/12/18/24 degrees along each horizontal axis, steps have 12/20/28 cm risers,
and uneven pad sets have maximum height differences of 6.5/18/24 cm.
Only reset joint/velocity perturbations are held out in the final seed: these
are familiar terrain presets, not evidence of unseen-terrain generalization.

Four downward ranges are real geometric ray queries, not a height-map lookup.
Existing angular velocity and gravity direction remain in the actor input; no
additional IMU is modeled. Contact bits and ideal body velocity/position error
are simulator feedback. Deploying this controller on hardware would require
corresponding estimators and separate validation.

The observer head camera is 40 cm forward and 8 cm above the base origin,
ahead of the nose mesh. The video also uses following and overhead views,
contact indicators and orange markers at actual limb removals. Observer floor
marks are render decorations only. Camera placement and encoder fixes do not
change physics, sensing or learned actions.

Historical training reports keep their original contents. Before the reporting
fix, `body_environment_counts` overwrote repeated healthy-terrain entries; use
the authoritative `standing_cases` list to recover all 4,096 worlds. One mixed
run reports `source_dirty=true`; this limitation is retained in its provenance.
