# Learning journal: from a utility selector to an embodied neural controller

This is the permanent account for future demos, videos and technical reporting.
Keep failed hypotheses as well as successful runs. The current project is in
progress; its complete flight/survival curriculum has not passed acceptance.

Project began **2026-09-12 18:05:28 UTC**. The approved architecture correction
began **19:35:03 UTC**. Clocked milestones are in [TIME_LOG](../TIME_LOG.md).
Development elapsed time, setup, data generation, optimization, physical evaluation
and rendering are different quantities and must be reported separately.

## What changed, and what we learned

| Stage | What we tried | What the evidence showed | Decision |
| --- | --- | --- | --- |
| Original survival lab | Eight physical FlyGym flies, needs, food/water, shelters, swatter, heat model, eyes and full frozen MaleCNS dynamics | A working physical/inspection scaffold; body movement came from FlyGym's supplied controller | Preserve arena/UI, but do not call this learned locomotion |
| Utility CEM 01 | Optimize 72 utility weights with eight CPU worlds | 80 episodes, 302.285 s optimization, zero numerical failures; ingestion behavior needed correction | Retain diagnostic; add local taste/ingestion and a stable intake hold |
| Utility CEM 02 | Repeat search after ingestion fixes | 96 episodes, 320.892 s optimization; all four held-out flies survived, but behavior camped at water and consumed no food | Survival alone was an insufficient success criterion |
| Neural ablation | Remove neural features from the CEM 02 evaluation | Mean score was essentially unchanged, 21.584 with features versus about 21.545 without | No demonstrated useful causal contribution from the frozen connectome |
| Utility CEM 03 | Balance needs and suppress irrelevant activity scores | Interrupted during the research pause; completed generations remain diagnostic | Do not turn an unfinished run into a result |
| Course correction | Train body actuation and utility inside one MaleCNS-based recurrent actor | User approved staged locomotion, recovery, wings, needs and social behavior | Replace the external gait dependency at student evaluation |
| Complete body | Preserve legs, wings, mouth and antennae in one physical model | FlyBody compiles with 108 DoF and 78 bounded actuators, free root, approximately 0.985 mg mass | Keep full action/observation schema across curriculum stages |
| Graph computation | Fixed measured adjacency with trainable cell dynamics and learned input/utility/motor interfaces | Dense-reference gradient checks, internal parameter changes, independent memory and a no-bypass test pass | Proceed to the full graph and GPU optimization |
| Teacher bridge | Convert the published learned walking policy to modern PyTorch inference | Numerical reference samples pass; physical complete-body holds and three speeds work in short tests | Use inherited teacher knowledge for supervised initialization only |
| Tight-turn demonstrations 01 | Hold/walk plus ±1.5 rad/s turns at all speeds | Three of sixteen two-second cases toppled: episodes 8, 9 and 12; zero numerical warnings | Separate physical failure from solver failure; retain rejected demonstrations |
| Demonstrations 02 | Holds/straight walking plus broad ±0.75 rad/s turns at ≥1 cm/s | Sixteen two-second Mac cases remained upright; GPU-workstation CPU collection repeated the recipe | Start imitation with this bounded motor curriculum |
| Full graph, eight sequences | Twenty-second CUDA optimization probe | Held-out motor MSE 0.65634 → 0.06988; internal core gradients and changes recorded | Full measured graph can be trained within a small pilot budget |
| Full graph, 32 sequences | Same short probe with larger neural batch | 32,512 supervised examples in 20.098 s; about 2.93 GB peak CUDA allocation | Use 32 sequences for the five-minute motor pilot |
| Motor BC 01 | Five-minute recurrent imitation, followed by teacher-free physical tests | Held-out MSE fell to 0.01630; **0/6 physical cases passed**, despite zero solver warnings | Retain failure; diagnose feedback/state-distribution mismatch before spending more training time |
| Passive-channel diagnostic | Same BC 01 checkpoint; set the 19 wing/mouth/antenna channels to the teacher's passive command, leaving 59 walking channels learned | All three paired hold/slow/walk cases became upright and had no disallowed ground support; **0/3 still passed command tracking** | Inactive appendage commands mattered; preserve this explicit walking curriculum while improving starts and feedback control |

Original CEM reports and checkpoints remain under
`docs/fly_survival/training/`, `docs/fly_survival/evaluation/` and
`assets/fly_survival/diagnostic_cem_*.npz`. Their source commits, seeds and hashes
are in those reports. The original gait controller and the tiny utility learner
are not the architecture of the new embodied actor.

## Exactly what the new actor learns

Current motor-stage schema:

```
383 causal body/command/need inputs
  → learned 383 → 128 → 11,238 sensory encoder
  → persistent state on 166,700 MaleCNS neurons
      25,582,938 fixed, signed, normalized connections
      learned excitability, leak and bias: 500,100 internal parameters
  → 1,314 descending states → 128 → six utility scores
  → selected intention encoded back into the same recurrent core
  → 815 motor-neuron states → 256 → 78 bounded actuator outputs
```

Layer normalization is used in the utility/motor readouts. There are **2,409,132
trainable parameters** in total. Four internal neural updates occur per motor
decision. State is a signed latent/rate-style computation with assumed dynamics,
not biologically calibrated voltage or measured spikes. The topology and neuron
identities come from the measured dataset. We are **not training 25.6 million
individual edge weights** in this version; the 500,100 trainable internal cell
parameters change computation inside that fixed topology.

Utilities remain an actual decision head. A winner feeds a learned intention
projection back into the graph before motor decoding. It does not call a walking
function. All actor parameters are saved together; each fly has independent memory.
The first warm start labels rest/explore from the commanded motor task. Feeding,
drinking, escape and grooming output slots exist, but are not acquired skills yet.

The student sees current joint state, actuator activation, local velocity and
orientation, foot contact, previous action, command and needs. Eye cameras are
rendered for observation at this stage; pixels are not student inputs yet. The
teacher alone receives the desired future COM path. Student evaluation imports
no teacher and supplies no gait phase or future trajectory.

## Measured compute: quantities that must stay distinct

| Run | Optimization | Parallel neural sequences | Supervised examples presented | Unique training frames | Initial → final held-out motor MSE | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Graph probe 01 | 20.023827 s | 8 | 9,728 | 12,000 | 0.656341 → 0.069877 | 1,295,849,472 bytes |
| Graph probe 32 | 20.097561 s | 32 | 32,512 | 12,000 | 0.649047 → 0.061284 | 2,930,399,232 bytes |
| Motor BC 01 | 300.033296 s | 32 | 484,864 | 12,000 | 0.649047 → 0.016298 | 2,930,399,232 bytes |

These probes use an **RTX 4090** for forward/backward passes. Eight and 32 refer
to independent demonstration sequences, **not live physical worlds**. Each update
uses eight context steps without gradient, then eight supervised recurrent steps.
The 12,000 unique training frames represent 24 seconds of recorded body experience;
repeated presentations do not create new physical experience. Validation uses
separate complete episodes. Its sampled batch differs between the two probes,
so the table is not a controlled comparison of final motor quality.

Training throughput was approximately 486 versus 1,618 supervised examples/s
including optimization. Full-graph setup took 3.884 s and 2.814 s respectively;
post-training validation took 0.085 s and 0.099 s. They are outside optimization
time. Whole-scene dynamics use **native CPU MuJoCo**, at 5 kHz physics and 500 Hz
motor decisions. MuJoCo Warp is an optional dependency and has not powered these
motor-learning runs. The earlier Warp fly hold was a separate physics-only probe.

Motor BC 01 performed **1,894 optimizer updates**. Setup took 2.764 s and its
offline post-training validation 0.099 s, outside the measured five-minute run.
The held-out imitation error reduction is about 97.5%, but **must not be presented
as a 97.5% improvement in walking**. In six two-second free-body evaluations,
five failed the tilt/height stability test; the fast-walk case stayed within that
stability envelope but failed speed, turning and allowed-support requirements.
All six failed overall, with no numerical failures. This is direct evidence that
offline action imitation is insufficient so far. The shared checkpoint is
`7bf16a8033a1013088e9672acbd9d44e4c08194c4ff8ddd744c270e9ae874243`.
Exact reports and curves are in [`runs/`](runs/); three diagnostic checkpoints are
retained under `assets/embodied_fly/diagnostics/` through Git LFS.

On the Mac, the four short teacher probes collected two physical seconds in
1.600 s after 2.922 s setup. Demonstration collection 01 collected 32 physical
seconds in 25.925 s after 2.929 s setup; collection 02 took 25.384 s after 2.918 s
setup. These are teacher/data costs, not training our graph. The GPU workstation
collected the 32-second dataset in **42.883 s after 6.142 s setup**, using CPU
physics on that machine. Do not substitute Mac numbers for its timing.

A subsequent replay diagnostic fed the frozen student a complete two-second
teacher observation sequence. MSE was 0.02009 overall, 0.06767 for the first 32
frames and 0.01703 for the last 500 frames. Neural-state standard deviation changed
from 0.1155 to 0.2087; this case did not show runaway recurrent-state error.
Predicted wing commands had normalized RMS 0.0323 despite a zero-wing-command
teacher target. This motivates testing passive wing/mouth/antenna channels during
the initial walking curriculum and training cold starts/feedback corrections.
The subsequent paired passive-channel diagnostic supported that concern: without
additional training, all three tested cases stayed upright (minimum vertical-axis
alignment ≥0.987) and had zero disallowed ground support. They still moved/turned
incorrectly for their commands, so none passed overall. The intervention muted
wings, mouth and antennae together; it does not isolate wing commands alone.
This is a declared first-stage actuator mask, not a new gait or a second policy.
All anatomy and degrees of freedom remain physical, and the actor still has 78
output slots for later curriculum stages. The mask must not be hidden in a claim
that flight or complete appendage control has already been learned.

The raw GPU captures remain in ignored run directories for re-rendering. Published
diagnostic reports/checkpoints are indexed with SHA-256 in
[`EXPERIMENT_INDEX.json`](EXPERIMENT_INDEX.json). This index also links the prior
utility experiments, so the narrative includes both architectures.

## Compatibility work and provenance

- FlyBody source is pinned at `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
  Preserve upstream physical parameters in CGS and convert reported metrics to SI.
- Python 3.14 installation encountered labmaze's missing wheel/Bazel build
  requirement. Excluding that dependency then failed dm-control arena import.
  Use compatible Python 3.12 with current MuJoCo 3.13.0, PyTorch 2.14.0 and NumPy
  2.5.3; keep the failed compatibility route documented rather than silently patched.
- Official teacher export uses isolated Python 3.13, TensorFlow 2.21.0 and current
  available TensorFlow Probability 0.25.0. Its old saved distribution type names
  need two explicit aliases during export. Exported weights include numerical
  reference inputs/outputs; runtime checks float32 agreement before use.
- Teacher weights came from the official Figshare data, whose record says
  GPL-3.0-or-later. Their original link, authors, checksum, conversion description
  and license are in `assets/embodied_fly/teachers/`. The body-code license is
  Apache-2.0. Do not attribute the teacher's original training time to our runs.
- Implementation was performed agentically through repository inspection,
  source/primary-paper research, coding, native tests, SSH/Git GPU execution and
  physical capture. Record the actual runtime/model identity if available;
  do not infer a specific model version from a draft social post.

## Reporting rules for films and demos

1. Show the source commit, checkpoint hash, data recipe and evaluation report
   behind each selected result. Save unsuccessful cases too.
2. Say **supervised imitation warm start** for these graph probes. They are not
   PPO or a demonstration of reward-driven survival learning. Later RL gets its
   own timed stage, while extending the same actor.
3. Falling, bad foot support and numerical failure are separate outcomes. Physical
   evaluation checks tilt, height, speed, turning and disallowed ground loads.
4. “Internal neural parameters learned” has gradient/weight evidence. “Measured
   topology improved learning” still requires a matched frozen/rewired control.
5. A successful teacher clip is labeled **reference teacher**. A low imitation
   error is not a successful free-body rollout, learned flight or reconstructed fly.
6. State whether a film is captured simulation, replay or generative styling, and
   identify observer-only camera feeds. Keep 1× unless another multiplier is shown.

The intended story is the learning process: build an embodied system, test a
hypothesis, retain its failure, change the architecture for a concrete reason,
then measure what the new controller actually learns.

### Feedback-data continuation plan (2026-09-12)

BC 01 remains the parent, with identical graph and 2,409,132-parameter actor.
The next short pilot introduces two explicit data/training changes: collect
teacher labels on states reached by a 25% student / 75% teacher actuator mixture,
and supervise the first eight reset frames in 25% of optimizer batches. Previously
every loss followed eight burn-in frames, leaving those initial decisions out.
The old demonstrations remain in the training pool, with whole-episode validation
splits preserved separately for each dataset. Observation normalization is retained
from the parent. BC 01 did not save Adam state, so the first continuation must
restart Adam; new checkpoints include optimizer state for subsequent resumes.

All 108 physical degrees of freedom remain. The declared walking curriculum holds
19 wing/mouth/antenna commands at raw zero; 59 channels come from the learned actor.
This is the exact intervention tested in the paired diagnostic, now shared by
collection and evaluation. No teacher may be used in student evaluation. Dataset
files distinguish teacher target actions from the executed physical mixture and
retain the parent hash, seed, collection time and failures. This is supervised
dataset aggregation, not PPO. The teacher still has a future path that the student
does not observe, so off-path target ambiguity remains a limitation to investigate
if this feedback-data pilot does not transfer.

### Tracking-frame correction before continuation

A check against the inherited teacher's actual displacement exposed a metric bug:
`mj_objectVelocity(..., mjOBJ_BODY, ..., local=1)` returns velocity in the body's
principal-inertia axes. The fly's inertia frame is rotated relative to its thorax.
Forward-speed and yaw metrics must use `mjOBJ_XBODY` (anatomical body axes).
The old BC01 reports are retained, but their speed/yaw values cannot establish
command accuracy. Tilt, height, support loads and numerical status are unaffected.
The initial three-case passive-mask result therefore proves stable support, with
command success requiring corrected evaluation. Existing trained velocity inputs
retain their original inertia coordinates; all six components remain available,
so no input reinterpretation or weight change is hidden in this evaluator fix.
A regression test applies known forward and yaw velocities at a rotated heading.
This distinction is explicit in the
[MuJoCo engine](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_util.c).

### Preserved interrupted utility search

CEM 03 was interrupted for the approved full-body architecture change. Its three
completed generations and intermediate checkpoints are now retained in
`docs/fly_survival/training/cem_utility_03_interrupted/` and the experiment index.
The last complete generation reports 231.865 seconds of training. Final total
time is unknown; partial work is not reconstructed or counted as a completed run.

### Feedback continuation 01: measured outcome

The RTX 4090 continuation completed **300.050483 s** of optimization after
**3.595310 s** setup, followed by **0.148990 s** offline validation. It performed
**2,059 updates**, including **523 reset-start batches**, and **527,104** supervised
frame presentations from **24,000 unique training frames (48 physical seconds)**.
Peak CUDA allocation was **2,930,873,344 bytes**. Twelve original episodes and
twelve mixture episodes trained the same actor; eight separate full episodes were
held out. Parent normalization was retained, Adam restarted because BC01 had not
saved its optimizer, and all three internal cell-parameter groups changed.
Checkpoint: `6e946635e0dbdcc1471529b573c8be4d6a7abc207b8fab7766530302cfa1c1d9`.
The selected ancestry has **600.083779 s** of optimization; separate initial
20-second probes and data/evaluation/rendering costs are additional work.

On the continuation's fixed held-out batch, motor MSE changed **0.013827 → 0.011609**;
reset-window MSE changed **0.185662 → 0.010461**. These are imitation errors, not
physical success. Both the original masked student and the continuation stayed
upright with zero prohibited support in all six two-second development cases.
The continuation's stop displacement decreased from approximately **15.9 mm to
8.5 mm**, still far too much for a stationary fly. Its right-turn behavior worsened.
No checkpoint is accepted as a complete locomotion solution.

The utility traces expose a shortcut: BC01 selected explore for every stop frame.
The continuation selected rest for **23.6%** of the stop trial, returning to explore
while moving. The next dataset therefore uses **100% student physical control**,
with teacher labels recorded but not executed, to provide rest targets on moving
states. It retains all previous data rather than replacing walking experience.
This is a further dataset-aggregation round, not a new policy or a scripted stop.

Teacher replay in anatomical axes also showed **0.82–4.84 rad/s raw yaw RMSE**,
despite accurate mean direction. Even the teacher fails the old 0.5 rad/s
instantaneous yaw threshold. The new tracking diagnostic reports both raw and
100 ms block-mean errors without replacing the recorded gates. The 1 cm/s teacher
example averages **1.0002 cm/s**, with block errors **0.0465 cm/s / 0.2377 rad/s**.
Raw jerk, sustained drift, tilt and support should remain separate quantities.
Replay metrics use captured pre-action poses/velocities; live reports sample after
the control step and can differ at substep phase. Compare checkpoints using the
same metric path and window.

Reports, learning curves, collection manifest, utility traces and the new student
film are preserved in `runs/motor_dagger_01/` and `previews/embodied_fly/`. The film
remains labeled as a motor-learning diagnostic.

### Feedback continuation 02: holding acquired, walking lost

The 100%-student collection ran **31.188 physical seconds** in **91.030721 s**
after **6.626658 s** setup. Fifteen episodes finished; episode 9 fell at 1.188 s.
Its failure and partial trajectory are retained, and that episode was excluded
from this imitation dataset. Teacher controls were recorded as labels, not
executed. A long replay of the teacher's original hold observations produced
**99.8% rest decisions** in the prior student, compared with **23.6%** during its
physical hold trial. That supports physical-state distribution shift as a problem,
without proving it is the only cause.

Continuation 02 preserved the actor, hyperparameters, old data, normalization and
Adam state. It used **300.009422 s** of optimization, **3.435756 s** setup and
**0.150242 s** offline validation, with **2,054 updates / 511 reset-start batches /
525,824 frame presentations**. There were **36,000 unique training frames (72 s)**
from 36 full episodes, with 11 other full episodes held out across the three data
pools. Peak CUDA allocation was **2,930,971,648 bytes**. Current reports bind every
episode file to SHA-256, including rejected data. Checkpoint:
`87384d048a946ac1175084538ac49a638518e06e607ec82521633104b9b7d102`.
Its ancestry contains **900.093201 s** of optimization; this is not a successful
15-minute complete-fly training claim.

Held-out action MSE changed **0.085706 → 0.051328**, but physical results regressed.
The hold case moved only **0.198 mm net** in two seconds with valid foot support.
The slow-walk command produced almost no travel. The walk, fast-walk and both
turns fell, with prohibited support. All six still failed the original aggregate
gates, and all had zero numerical warnings. The earlier walking checkpoint remains
available; do not stitch these checkpoints into a claim of one successful policy.
This continuation is retained as a failure to preserve skills.

### Receding teacher diagnostic, not adopted as a solution

A new optional `--reference-mode receding` anchors the teacher's command-derived
lookahead to current xy/heading, removing its request to return to an absolute
path invisible to the student. It leaves the physical body untouched and keeps
the intended standing height. A regression test verifies invariance to global
translation/heading and no pose/time mutation. This is training-only reference
construction, not a deployed movement controller.

Sixteen two-second Mac teacher probes took **29.665335 s** collection after
**2.991828 s** setup; all stayed upright. However, low-speed tracking and some turns
were poor: at a 0.5 cm/s command, measured mean speed ranged about **0.077–0.266
cm/s**. One 1 cm/s negative-turn case turned the wrong way. The diagnostic is saved
under `runs/receding_teacher_probe_01/`; it has not trained an accepted student.
It does not justify another blind extension of the same imitation recipe.

The next implementation stage is the already approved physical-outcome RL stage:
start from a retained walking-capable checkpoint, reward actual command tracking,
stationary hold and valid support, and preserve learned stepping through explicit
rehearsal. Keep the same deployed graph actor and learned utility head. A separate
training-only critic is allowed; no teacher, gait generator or action selector may
replace the actor at evaluation. Native/GPU throughput and recurrent rollout
requirements must be measured before selecting the live world count.

### Actual neural-activity video overlay

The optional observer capture bins all **140,638 located neurons** into a cached
128 × 128 map at 50 Hz; **26,062 cells without locations remain in computation**.
Each bin records mean signed and mean absolute recurrent latent activity. Fixed
amber/teal colors show sign; these are not physiological spikes. The unit test
checks independent worlds, sign/magnitude averaging and unchanged source state.
The overlay does not change the actor, observations or rewards.

Two separate diagnostic films show BC01 walking and DAgger02 holding with this
overlay. Each is **2 s / 100 frames / 50 fps / 1×**, fully decoded and visually
inspected. Rendering took **7.857275 s** and **8.065591 s** on the GPU machine.
Their source/checkpoint/state/model hashes are retained. An initial setup rejection
of legacy BC01's absent metadata digest is also recorded; the loader now checks
exact neuron routing and graph weights for legacy files, and additionally checks
metadata digests when stored. No simulation ran in that rejected setup.

The user reconfirmed that flight remains required. Full-body wing control, takeoff,
flight, landing, learned survival utility, multi-agent integration and the final
combined demonstration are still open. These prototypes do not complete that goal.

### Native live-physics batching for outcome learning

Added a separate `FlyBatch` using the repository's pinned, Apache-2.0 mjbatch
source (revision `77966f85bcd8f7ef4351cb4a1a6f42e133d19725`, MuJoCo 3.13 build).
It retains full anatomy, passive-channel walking curriculum, all force bounds,
500 Hz control / 5 kHz physics, and the existing 383-value observation convention.
Additional sensors expose anatomical/inertial-frame velocity, claw contacts and
forbidden support. Reward support samples every physics substep; authoritative
acceptance still enumerates every contact in the original evaluator. The training
sensor selects the maximum-force-norm contact per geom, so is explicitly a proxy
for the acceptance maximum over every normal reaction.

A 100-action regression test produced identical single/batched poses and float32
observations, matching anatomical velocities and forbidden contact loads. It also
checks independent worlds and repeated partial resets. The first implementation
lost the folded-wing initialization on repeated resets because mjbatch only
applies changed array elements. Reset-then-initialize-then-forward fixes this;
the regression includes the previously failing repeated-reset case. No training
used that defective implementation. Physics remains CPU-based; CUDA is reserved
for the full-graph actor and learning in this path.

### First physical-outcome PPO recipe (declared before the run)

Start from **motor_dagger_01**, which retained upright, permitted-foot support in
all six command cases and partially learned rest. The later DAgger02 checkpoint
lost four locomotion cases and is not this warm start. Keep the complete measured
graph, 2,409,132 actor parameters, 383 causal inputs, 78 output channels, four
internal recurrent updates per 2 ms action, and the walking-stage 19-channel
passive mask. Graph neuron dynamics and both utility/motor interfaces receive
physical-reward gradients. No teacher executes actions during PPO collection.

Use 32 independent native CPU worlds with 16 threads and CUDA neural computation.
The physics-only 1/8/16/32-world probe measured approximately 588/3,442/5,021/6,159
transitions/s, including sensor extraction; those numbers exclude neural inference
and optimization. The initial PPO rollout is 64 actions × 32 worlds, two epochs,
eight-step recurrent chunks, gamma 0.995, GAE lambda 0.95, learning rate 1e-5,
ratio clip 0.2 and KL early stopping at 0.03. Initial motor noise is 0.04 in
pre-tanh action coordinates. Utility is sampled from the same six learned scores;
it conditions the same graph before the motor readout. PPO uses the joint
categorical-plus-squashed-Gaussian probability, without a straight-through
categorical gradient. The unchanged greedy actor is used for evaluation.

A separate training-only critic consumes normalized causal observations and the
preceding descending-neuron state: 1,697 → 128 → 128 → 1. It estimates return,
not rewards, and does not drive any actuator. Explicit task code adds rate rewards
for measured velocity/yaw tracking, upright support, and costs for tilt, vertical
motion, action changes and prohibited support. Rates are multiplied by 0.002 s;
a physical fall adds a one-time −1 and resets that world. Command tracking uses a
50 ms velocity filter to avoid equating natural within-stride yaw with command
drift. The original raw evaluation gates remain unchanged and visible.

Hold, slow/normal/fast walking and both turns coexist in the same world batch.
Episodes last up to two seconds. Timeouts bootstrap the terminal observation's
value; falls do not. Neither transition leaks value or recurrent memory from the
next episode. Chunk boundaries retain the actual preceding rollout state; memory
is cleared only at true episode resets. A short original-teacher-data rehearsal
batch follows each rollout, with the same held-out episode split, 8-step context,
25% reset contexts, MSE + 0.02 utility CE. Its compute and frame count are reported
separately and also included in total training wall time. This is an explicit
recipe change from pure imitation, not a continuation with identical optimizer.

Fourteen focused tests pass, including exact recurrent sampled-action replay,
physical-reward gradient flow into internal cells, timeout/failure return handling,
and native/single-body physical agreement. First run is a short implementation
pilot; a longer trial depends on its measured behavior and numerical health.

### Outcome PPO implementation pilot 01

The 20-second requested pilot completed one final rollout at **21.075403 s**,
collecting **30,720 physical transitions / 61.44 aggregate simulated seconds**.
Setup took **4.765655 s**. Collection (CPU physics + CUDA inference + bookkeeping)
took **15.905340 s**; PPO optimization **3.028985 s**; explicit rehearsal
**2.131341 s / 3,840 frame presentations**. There were 16 PPO minibatch updates
across 15 rollouts, 32 live worlds and 16 CPU physics threads, approximately
**1,458 transitions/s including learning**, with **5,342,384,640 bytes** peak CUDA
allocation. The training-only critic has 233,985 parameters. Physical-reward
gradients reached over 165,000 cells for each of excitability, leak and bias, with
finite nonzero gradients and measurable parameter changes. This proves gradient
flow through the measured graph; it does not yet prove useful brain causality.

Two collection worlds fell; most episodes had not yet reached their two-second
limit when this short run ended. Do not treat completed-episode counts as a
success denominator. That pilot retained failure metrics but not training-fall
trajectories. The next version stores the last up to 128 action frames (0.256 s)
for every training fall, with full physical states/actions/utility and a model
hash, and saves an error snapshot for numerical exceptions. Full six-case
teacher-free evaluation captures already retain all states, including failures.

The greedy pilot checkpoint stayed upright with valid support in all six cases,
but still failed the original raw tracking gates. Normal walking's 100 ms
block-mean forward error was **0.093 cm/s**, and yaw error **0.435 rad/s**. Hold
still drifted about **10 mm net in two seconds**; the right turn remained weak.
These are diagnostics, not integrated behavior acceptance.

The initial Adam learning rate 1e-5 caused KL excursions (first approximately
0.282, versus the 0.03 early-stop threshold), allowing only one or two PPO chunks
per rollout. For the next **300-second trial**, reduce only the actor/exploration
learning rate to **2e-6** and use seed 28002. Start again from the same DAgger01
parent, not from the pilot, preserving the pilot as a separate experiment. Retain
reward weights, recurrent architecture, 32 worlds, 16 threads, rollout/rehearsal
recipe and evaluation cases. This is an explicitly documented step-size adjustment
based on the measured update size. It has not changed the acceptance gates.
