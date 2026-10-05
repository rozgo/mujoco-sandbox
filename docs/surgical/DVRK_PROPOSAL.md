# Surgical robotics in MuJoCo with PufferLib 5.0

Research and proposal, October 5, 2026. No surgical simulation or training has run.
The recommendation is **a dual-arm dVRK Si system**, beginning with rigid-object
needle skills and growing into thread handling and compliant-pad closure.

## Robot selection

| Candidate | Available foundation | Assessment for this project |
| --- | --- | --- |
| **dVRK Si: two patient-side manipulators (PSMs)** | Official URDF/Xacro, meshes, replaceable instrument definitions, some CAD-derived inertias | Best surgical fit; requires a maintained MuJoCo conversion and dynamics validation |
| **Two Franka Panda arms** | Existing Menagerie MJCF, actuators and grippers | Fastest ready-made MuJoCo fallback; surgical shafts, wrists and port constraints would be custom additions |
| **Raven II** | Established open surgical research platform | Credible alternative, but this investigation did not establish a better maintained, ready-to-run MuJoCo asset than the dVRK route |
| **Existing Kinova Gen3 assets** | Already composed into this repository's manipulation demo | Useful adapter/debug fixture; ordinary arm and gripper geometry does not supply a surgical instrument mechanism |

This is a recommendation for the best **surgical foundation to port**, not a claim
that dVRK is already a validated MuJoCo download. The current Menagerie inventory
has Panda, FR3 and Kinova models; its dVRK addition request remains open.
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
[dVRK request](https://github.com/google-deepmind/mujoco_menagerie/issues/241),
[Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda),
[Raven authors' overview](https://arxiv.org/abs/1906.11747).

The official dVRK source supplies Classic and Si assemblies, camera arms, and
instrument geometry. Use Si needle driver **420006**, whose tip geometry is present;
several other instrument entries have placeholder tips. Preserve the MIT license
and upstream instrument notices. This is a dVRK Si research assembly, not a digital
twin claim for a current commercial da Vinci system.
[Official assets](https://github.com/jhu-dvrk/dvrk_model).

Source inspection found CAD-derived mass/inertia entries in the Si PSM base, but
also zero effort/velocity limits and a 1000-valued insertion effort field. These
are not accepted as calibrated actuator specifications. The Classic base inspected
has mimic linkages and no inertial entries. Si therefore offers a useful starting
point, with actuator identification and parameter accounting still required.
[Inspected Si source](https://github.com/jhu-dvrk/dvrk_model/blob/fa39a466ff2c8d2a33637433ae950555172bcafa/urdf/Si/PSM_base.urdf.xacro),
[Classic source](https://github.com/jhu-dvrk/dvrk_model/blob/fa39a466ff2c8d2a33637433ae950555172bcafa/urdf/Classic/PSM_base.urdf.xacro).

SurRoL and ORBIT-Surgical are useful surgical-task references. Their existing
PyBullet/Isaac ecosystems are not drop-in MuJoCo/PufferLib environments. The
official dVRK simulator list also documents other engines; this proposal adds a
distinct MuJoCo implementation.
[SurRoL](https://github.com/med-air/SurRoL),
[ORBIT-Surgical](https://orbit-surgical.github.io/),
[dVRK simulator documentation](https://dvrk.readthedocs.io/main/pages/software/simulators.html).

## How complex the robot can become

The proposed first robot has two instrument arms, each with six tool-pose axes
plus jaw opening: **14 independent commands**. A four-axis camera manipulator
brings the assembly to **18**. A third instrument arm raises it to **25**.
Passive linkage joints and deformable-object coordinates increase the physical
state beyond these command counts. These counts describe this assembly, not a
MuJoCo capacity limit. Initially the camera is held by a reference controller.

Model the actual remote center of motion (RCM): the instrument shaft pivots about
the entry port. Keep the parallelogram and jaw couplings physically constrained,
using explicit linkages/equality constraints or a separately documented equivalent
mechanism whose kinematics and inertia have been checked. Do not treat each visual
link as an independently actuated joint. Use real base mounts, finite force limits,
and physical opposing-jaw contact to hold the needle.

Planned scene: mounted arms surrounding a compact benchtop trainer, trocar/port
guides, a needle tray, ring targets and a receiving cradle. Later variants add
thread, a compliant pad, a retraction tool, and a mechanically driven moving
support to emulate motion of the task surface. Setup-joint/cart motion stays fixed
during the initial task. Tool changing and learned camera positioning are later
tasks with their own mechanics and tests.

The fidelity limit is primarily **contact and tissue mechanics**, rather than the
number of robot joints. MuJoCo provides deformable flex elements and an elastic
cable plugin. Those are useful ingredients for pad and thread prototypes. They do
not by themselves validate puncture, fracture, knot security, or biological tissue
response. Native tendon paths also should not be confused with a freely colliding
thread. [MuJoCo deformables and cables](https://mujoco.readthedocs.io/en/stable/modeling.html#deformable-objects).

## Tasks and a concrete flagship

| Stage | Complete task | Main difficulty |
| --- | --- | --- |
| Plant validation | Hold, reach, close jaws, lift and release a needle | RCM, limits, tiny collision geometry and stable friction |
| First learned mission | Pick up a curved needle, orient it, hand it to the other arm, pass through two rings, place it in a cradle | Two-arm coordination, changing grasps, sequence memory and constrained passage |
| Threaded mission | Perform the passage with attached thread, pull slack through and maintain a tension band | Cable contact, self-contact, snagging and tension estimation |
| Flagship extension | Close a split compliant pad through paired pre-existing eyelets, alternating hands, then release with closure retained | Long sequences, compliant contact, tension and retention mechanics |
| Research extension | Needle puncture, continuous stitch placement, knot tying, camera-based operation and disturbance recovery | Material failure, thread/tissue coupling, topology and partial observation |

The first learned mission should occupy roughly 20–60 simulation seconds, subject
to measured reference-controller timing. It should look like a continuous surgical
skills exercise: approach, grasp, rotate, transfer, thread the rigid needle through
targets, and place. Regrasping and recoverable slips can be added after nominal
completion is repeatable. Peg transfer is a simpler diagnostic curriculum step.

The ambitious visual target is **bimanual wound-closure practice on a moving soft
pad**. Begin with actual modeled eyelets and disclose them; eyelet threading is not
evidence of tissue puncture. A claimed knot must survive a physical retention test
after both grippers release. Cable crossing in a render is insufficient. Realistic
cutting, cautery and bleeding would require additional coupled models and material
validation; they are outside the initial implementation estimate.

## PufferLib 5.0 integration

The current 5.0 path uses a C/CUDA trainer and native environment hooks. Its docs
state that CPU evaluation is available but CPU training is not. Plan Linux/NVIDIA
training and local CPU evaluation. Older Python Gym-wrapper recipes are not the
implementation plan for this version.
[PufferLib 5.0 documentation](https://puffer.ai/docs.html).

Direct inspection confirmed Gaussian continuous actions and rejection of mixed
continuous/discrete action spaces at the inspected revision. Make jaw aperture a
continuous output alongside arm commands. The native API exposes initialization,
reset, step, render, close and logging hooks, with observation/action/reward/terminal
buffers. These interfaces support the proposed adapter; they do not constitute an
already implemented MuJoCo binding.
[Trainer source](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferl.cu),
[Environment interface](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferenv.h).

Proposed first implementation:

1. **One recurrent actor controls both arms.** Start with 14 continuous outputs:
   six bounded tool-motion increments plus jaw opening per arm. A constrained
   low-level controller turns those commands into force-limited actuator targets.
   Label this as learned task control with a programmed servo controller.
2. **Native C/C++ MuJoCo core.** Allocate independent `mjData` per world and
   read-only shared models where parameters match. Use per-world model copies or
   immutable variant pools for physical randomization. Never mutate shared model
   parameters from parallel workers.
3. **Puffer native adapter.** Implement the `puf_*` lifecycle and link `libmujoco`.
   Run physics, feedback, observations, events and rewards in the native step.
   Reuse that core in the Python/Mac inspector through a thin binding.
4. **CPU physics, GPU policy updates initially.** Start with 32–128 worlds, then
   benchmark scaling toward 256–512. These are experiment sizes, not throughput
   promises. Avoid nested thread pools under Puffer's environment workers.
5. **Separate clocks.** Trial 1 ms physics, 500–1000 Hz servo feedback, 50–100 Hz
   policy commands and 30–60 Hz viewing. Confirm convergence at smaller timesteps
   on needle/gripper contacts before fixing the contract. Values are starting
   assumptions, not established stability results.
6. **State observations first.** Tool/joint state, object-relative pose, task goal,
   previous commands, contact features and remaining time. Keep exact simulator
   object state explicitly labeled. Add endoscopic vision only in a separate
   measured stage with an encoder, rendering budget and observation audit.

Use an explicit fixed deadline as a terminal condition with remaining time in the
observation for the first finite-horizon task. Test reset and recurrent-state
boundaries. The inspected environment API exposes a terminal buffer without a
separate Gym-style truncation buffer; do not silently map arbitrary interrupted
rollouts to task failure. Audit action transformation/log-probability consistency
and compare Puffer's default reduced precision with its float mode for fine motion.

A reference controller establishes feasibility and records demonstrations before
RL. Start PPO with staged initial-state difficulty and rewards based on physical
progress. If sparse needle skills stall, add explicitly labeled imitation or a
demonstration loss; that is additional trainer work, not an assumed turnkey 5.0
feature. Use one actor for coordinated manipulation before considering separate
agents or a learned high-level task selector.

MuJoCo Warp could later increase simulation throughput. Its Python/Warp execution
and Puffer's native runtime need explicit interoperability, buffer and stream work.
First verify required constraints, deformables and plugin behavior, then compare
against the CPU reference. GPU policy learning alone does not make physics GPU
based. [MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html).

## Setup and repository layout

Use an isolated surgical worktree/branch for implementation. Existing source already
has model composition, macOS `mjpython` launching, recording, and CPU batch patterns
worth reusing; the existing Python `mjbatch` wrapper is not itself a native Puffer
adapter. Keep the common dynamics/controller implementation single-sourced.

Proposed paths, not created implementation:

```text
experiments/surgical/              isolated uv project, lockfile, native build
  src/surgical_robot/              composition, viewer, capture and reports
  native/                         environment, controller, Puffer adapter
  configs/                        scene contracts, curriculum and evaluation
  tests/                          physics, interface and complete missions
assets/surgical/dvrk/              pinned source assets, licenses and hashes
assets/surgical/checkpoints/       selected learned policies and manifests
docs/surgical/                    brief, setup, measured results
previews/surgical/                selected scene images and reviewed videos
outputs/surgical/                ignored trajectories, logs and checkpoints
build/surgical/                  ignored generated MJCF and native builds
```

Suggested CLI name: `surgical-robot`, exposing preview, view, run, record, train
and evaluate. Commands will be documented and smoke-tested once implemented; no
working surgical viewer command exists at this research checkpoint.

| Resource | Proposed setup |
| --- | --- |
| Mac | Existing uv/MuJoCo workflow for scene generation, tests, viewer and CPU inference |
| Training host | Linux workstation, one NVIDIA GPU with 24–32 GB VRAM, 16–32 CPU cores, 64 GB RAM as an initial sizing recommendation, not a measured minimum |
| Python tooling | Isolated Python 3.12 uv project, pinned MuJoCo, NumPy, mesh/Xacro tooling, test and media utilities |
| Native tooling | C/C++ compiler, OpenMP, MuJoCo headers/library and pinned Puffer 5.0 source |
| CUDA tooling | Development toolkit with `nvcc` and Puffer's CUDA/NCCL dependencies; pin a tested container and compatible host driver |
| Storage | Provision roughly 50–100 GB for initial assets, builds, state runs and selected media; vision datasets require a separate budget |

The inspected PufferTank container starts from CUDA **13.0.2 cuDNN development /
Ubuntu 24.04**. This is the observed reference image, not a claim that every Puffer
installation requires that exact version. Inspect/pin its build instead of running
an unreviewed global installer on the Mac. Python dependencies remain in uv; native
and CUDA dependencies belong in the build/container specification.
[Pinned container](https://github.com/PufferAI/PufferTank/blob/240d9794872fc1f9e0948588d1f453dbbc15b8f5/puffertank.dockerfile),
[Installer dependencies](https://github.com/PufferAI/PufferTank/blob/240d9794872fc1f9e0948588d1f453dbbc15b8f5/install.sh).

No physical robot or full ROS installation is necessary for the proposed runtime.
Xacro expansion/package resolution can be confined to asset preparation. Hardware
integration would add independent calibration, communication and validation work.

## Gates and estimated effort

These are engineering planning ranges for one experienced developer, assuming
usable upstream geometry and access to a GPU host. They are not measured run times
or guaranteed completion dates.

| Milestone | Estimated effort | Evidence required |
| --- | --- | --- |
| Static two-arm scene | 1–3 working days | Compile; inspect overview, jaw/needle detail and port clearance before movement |
| Validated physical plant and native adapter | 1–2 further weeks | Feasible reference mission; force-limited contact grasps; native/single-world agreement; continuous-action training smoke test |
| Reliable learned bimanual needle mission | 2–4 further weeks | Held-out success, physical limits and failures reported across training seeds; matched 1x reference/policy video |
| Thread and compliant-pad closure | 1–3 further months | Stable cable and pad mechanics, tension/closure metrics, physical retention and recovery tests |
| Tissue puncture/cutting or robust visual autonomy | Separately scoped research | Validated material laws/perception and revised compute plan before a schedule commitment |

Compute timing is unmeasured. First measure physics substeps/s, full agent
transitions/s, inference, updates, memory and rendering separately. For example,
10 million agent transitions take 2.78 hours at an actually measured end-to-end
1,000 transitions/s, or 16.7 minutes at 10,000 transitions/s; this arithmetic is
not a predicted surgical benchmark. Pilot budgets should expand only after a
learning curve shows repeatable progress. Puffer arcade throughput and results
from a different surgical simulator cannot predict this task's training time.

Proposed initial acceptance thresholds, to freeze after plant validation and before
policy tuning:

- Nominal held-out mission success at least **90/100 episodes per training seed**,
  for three predetermined seeds. Report counts and confidence intervals, including
  every failure and timeout. Maintain a separate randomized stress suite.
- Successful mission includes a contact-supported lift, receiver grasp established
  before donor release, geometrically verified passage through both ring openings,
  and stable final release into the cradle. No pose writes or grasp welds.
- Proposed final needle pose tolerance: **2 mm / 5 degrees**, followed by **0.5 s**
  at less than **5 mm/s** linear and **5 degrees/s** angular speed after release.
- Proposed peak RCM deviation **0.5 mm**; reject forbidden arm/fixture contacts,
  nonfinite states and MuJoCo warnings. Set mesh-scale penetration limits from
  needle/jaw dimensions and timestep convergence, rather than an arbitrary large
  tolerance.
- Measure contact loads and actuator saturation at every physics substep. Set
  numerical force limits after the parameter ledger and reference tests; no
  patient-safe force threshold is inferred from these models.
- Audit reward/termination ordering with successful, slow-but-valid, recoverable,
  and failed trajectories. Added effort penalties must not reward early failure.

Record physical-contract hashes, source commit, backend, seeds, policy, action
mapping, states, controls, forces and task events. Camera feeds are synchronized
observer output in the state-based baseline. Inspect encoded key frames and decode
the complete final MP4; open it on the Mac after verification. A separate visual
demo does not replace held-out evaluation.

## Research provenance and next action

Source snapshots inspected:

- dVRK model: `fa39a466ff2c8d2a33637433ae950555172bcafa`.
- PufferLib, branch 5.0: `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.
- PufferTank, branch 5.0: `240d9794872fc1f9e0948588d1f453dbbc15b8f5`.
- MuJoCo Menagerie: `4d038b3feae26ec82b46a4d586379114012a8ac7`.

Checked repository documentation and existing scene/viewer/batch code. External
source inspection is not a build or runtime test. No dependencies were installed,
no GPU job was submitted, and no assets were promoted as validated robot models.

**Next implementation action:** convert one Si PSM with the 420006 needle driver,
audit its frames/inertias/limits and render a static port-and-needle scene. Duplicate
it into the two-arm layout only after those checks. Continue to dynamics when the
RCM, jaw geometry and contact clearances are credible. If the asset cannot meet
those checks within the initial investigation, revise the conversion or use a
clearly labeled Panda/Kinova fixture to validate the Puffer adapter independently.
